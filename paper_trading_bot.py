"""
Paper Trading Bot (dinero ficticio) con CCXT + Streamlit
--------------------------------------------------------
Ejecutar:  streamlit run paper_trading_bot.py

Estrategia:
  COMPRA : el RSI(14) cruza hacia arriba el nivel 30 (sale de sobreventa)
           Y el precio de cierre está por encima de la SMA(50).
  VENTA  : Take Profit +2%  o  Stop Loss -1.5% (sobre el precio de entrada).

Solo usa datos públicos del exchange: NO necesita API keys ni mueve dinero real.
"""

import json
import os
import time
from datetime import datetime, timezone

import altair as alt
import ccxt
import pandas as pd
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# CONSTANTES
# ----------------------------------------------------------------------------
STATE_FILE = "paper_state.json"
INITIAL_CAPITAL = 100.0  # USDT ficticios
RSI_PERIOD = 14
SMA_PERIOD = 50
RSI_OVERSOLD = 30
TAKE_PROFIT_PCT = 2.0
STOP_LOSS_PCT = 1.5
EXCHANGES = ["kraken", "binanceus", "bybit", "okx", "binance"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]


# ----------------------------------------------------------------------------
# ESTADO PERSISTENTE (sobrevive a recargas del navegador)
# ----------------------------------------------------------------------------
def default_state() -> dict:
    return {
        "cash": INITIAL_CAPITAL,
        "qty": 0.0,
        "entry_price": 0.0,
        "trades": [],
        "last_signal_ts": None,
    }


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            base = default_state()
            base.update(data)
            return base
        except (json.JSONDecodeError, OSError):
            pass
    return default_state()


def save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError as e:
        st.warning(f"No se pudo guardar el estado: {e}")


# ----------------------------------------------------------------------------
# DISCORD
# ----------------------------------------------------------------------------
def send_discord(webhook_url: str, message: str) -> None:
    if not webhook_url:
        return
    try:
        requests.post(webhook_url, json={"content": message}, timeout=10)
    except requests.RequestException as e:
        st.warning(f"No se pudo enviar la alerta a Discord: {e}")


# ----------------------------------------------------------------------------
# DATOS DE MERCADO E INDICADORES
# ----------------------------------------------------------------------------
@st.cache_resource
def get_exchange(exchange_id: str):
    return getattr(ccxt, exchange_id)({"enableRateLimit": True})


def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    ex = get_exchange(exchange_id)
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms")
    return df


def fetch_price(exchange_id: str, symbol: str) -> float:
    ex = get_exchange(exchange_id)
    return float(ex.fetch_ticker(symbol)["last"])


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI de Wilder."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100.0).where(avg_gain.notna())


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"])
    df["sma50"] = df["close"].rolling(SMA_PERIOD).mean()
    return df


# ----------------------------------------------------------------------------
# OPERACIONES SIMULADAS
# ----------------------------------------------------------------------------
def now_str() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def do_buy(state: dict, price: float, fee_pct: float, symbol: str, webhook: str, reason: str) -> None:
    invest = state["cash"]
    if invest <= 0:
        return
    qty = invest * (1 - fee_pct / 100) / price
    state["qty"] = qty
    state["entry_price"] = price
    state["cash"] = 0.0
    state["trades"].append(
        {"time": now_str(), "side": "COMPRA", "price": price, "qty": qty, "reason": reason, "pnl": None}
    )
    save_state(state)
    send_discord(
        webhook,
        f"🟢 **COMPRA** {symbol}\nPrecio: {price:,.2f}\nCantidad: {qty:.6f}\nInvertido: {invest:,.2f} USDT\nMotivo: {reason}",
    )


def do_sell(state: dict, price: float, fee_pct: float, symbol: str, webhook: str, reason: str) -> None:
    if state["qty"] <= 0:
        return
    qty = state["qty"]
    proceeds = qty * price * (1 - fee_pct / 100)
    cost_basis = qty * state["entry_price"]
    pnl = proceeds - cost_basis
    pnl_pct = (price / state["entry_price"] - 1) * 100
    state["cash"] = proceeds
    state["qty"] = 0.0
    state["entry_price"] = 0.0
    state["trades"].append(
        {"time": now_str(), "side": "VENTA", "price": price, "qty": qty, "reason": reason, "pnl": pnl}
    )
    save_state(state)

    if reason == "Stop Loss":
        icon, title = "🛑", "STOP LOSS ACTIVADO"
    elif reason == "Take Profit":
        icon, title = "💰", "VENTA (Take Profit)"
    else:
        icon, title = "🔴", "VENTA"
    send_discord(
        webhook,
        f"{icon} **{title}** {symbol}\nPrecio: {price:,.2f}\nResultado: {pnl:+,.2f} USDT ({pnl_pct:+.2f}%)\n"
        f"Saldo total: {state['cash']:,.2f} USDT",
    )


def run_bot_cycle(state, df, live_price, fee_pct, symbol, webhook):
    """Una pasada del bot: revisa TP/SL con el precio en vivo y busca señal de compra en velas cerradas."""
    # 1) Gestión de posición abierta (TP / SL con precio en vivo)
    if state["qty"] > 0:
        entry = state["entry_price"]
        if live_price >= entry * (1 + TAKE_PROFIT_PCT / 100):
            do_sell(state, live_price, fee_pct, symbol, webhook, "Take Profit")
            return
        if live_price <= entry * (1 - STOP_LOSS_PCT / 100):
            do_sell(state, live_price, fee_pct, symbol, webhook, "Stop Loss")
            return
        return  # posición abierta: no buscar nuevas compras

    # 2) Señal de compra sobre velas CERRADAS (se descarta la vela en curso)
    closed = df.iloc[:-1].dropna(subset=["rsi", "sma50"])
    if len(closed) < 2:
        return
    last, prev = closed.iloc[-1], closed.iloc[-2]
    if state["last_signal_ts"] == int(last["ts"]):
        return  # esta vela ya fue evaluada

    rsi_cross_up = prev["rsi"] < RSI_OVERSOLD <= last["rsi"]
    above_sma = last["close"] > last["sma50"]
    state["last_signal_ts"] = int(last["ts"])
    save_state(state)

    if rsi_cross_up and above_sma:
        do_buy(
            state, live_price, fee_pct, symbol, webhook,
            f"RSI cruzó {RSI_OVERSOLD} al alza ({prev['rsi']:.1f} → {last['rsi']:.1f}) y precio > SMA{SMA_PERIOD}",
        )


# ----------------------------------------------------------------------------
# INTERFAZ STREAMLIT
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Paper Trading Bot", page_icon="📈", layout="wide")
st.title("📈 Paper Trading Bot — dinero ficticio")

if "state" not in st.session_state:
    st.session_state.state = load_state()
state = st.session_state.state

# ---- Barra lateral ----
with st.sidebar:
    st.header("⚙️ Configuración")
    exchange_id = st.selectbox("Exchange (datos públicos)", EXCHANGES, index=0)
    symbol = st.text_input("Par", value="BTC/USDT")
    timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=3)
    fee_pct = st.number_input("Comisión simulada por operación (%)", 0.0, 1.0, 0.1, 0.01)
    refresh_s = st.slider("Actualizar cada (segundos)", 5, 120, 15)
    webhook = st.text_input(
        "Webhook de Discord",
        value=os.getenv("DISCORD_WEBHOOK_URL", ""),
        type="password",
        help="También puedes definir la variable de entorno DISCORD_WEBHOOK_URL.",
    )
    running = st.toggle("🤖 Bot activo", value=True)

    c1, c2 = st.columns(2)
    if c1.button("Probar Discord"):
        send_discord(webhook, "✅ Prueba: el bot de paper trading está conectado.")
        st.toast("Mensaje enviado (si el webhook es válido).")
    if c2.button("♻️ Reiniciar"):
        st.session_state.state = default_state()
        save_state(st.session_state.state)
        st.rerun()

    st.caption(
        f"Compra: RSI{RSI_PERIOD} cruza >{RSI_OVERSOLD} y precio > SMA{SMA_PERIOD}\n\n"
        f"Venta: TP +{TAKE_PROFIT_PCT}% / SL -{STOP_LOSS_PCT}%"
    )

# ---- Datos + ciclo del bot ----
try:
    df = add_indicators(fetch_ohlcv(exchange_id, symbol, timeframe))
    live_price = fetch_price(exchange_id, symbol)
except Exception as e:  # red, símbolo inválido, exchange bloqueado, etc.
    st.error(f"No se pudieron obtener datos de {exchange_id} para {symbol}: {e}")
    st.info("Prueba con otro exchange o revisa el símbolo (ej. BTC/USDT).")
    st.stop()

if running:
    run_bot_cycle(state, df, live_price, fee_pct, symbol, webhook)

# ---- Métricas ----
position_value = state["qty"] * live_price
total_balance = state["cash"] + position_value
pnl_usd = total_balance - INITIAL_CAPITAL
pnl_pct = pnl_usd / INITIAL_CAPITAL * 100
color = "#16a34a" if pnl_usd >= 0 else "#dc2626"

m1, m2, m3, m4 = st.columns(4)
m1.metric("💵 Saldo Total Virtual", f"{total_balance:,.2f} USDT")
with m2:
    st.markdown("**P&L (Ganancia / Pérdida)**")
    st.markdown(
        f"<div style='font-size:2rem;font-weight:700;color:{color};'>"
        f"{pnl_usd:+,.2f} USDT<br><span style='font-size:1.2rem;'>({pnl_pct:+.2f}%)</span></div>",
        unsafe_allow_html=True,
    )
m3.metric(f"Precio {symbol}", f"{live_price:,.2f}")
if state["qty"] > 0:
    unreal = (live_price / state["entry_price"] - 1) * 100
    m4.metric("Posición abierta", f"{state['qty']:.6f}", f"{unreal:+.2f}% vs entrada")
else:
    m4.metric("Posición abierta", "Ninguna", f"Efectivo: {state['cash']:,.2f}", delta_color="off")

if state["qty"] > 0:
    entry = state["entry_price"]
    st.caption(
        f"Entrada: {entry:,.2f} · Take Profit: {entry * (1 + TAKE_PROFIT_PCT / 100):,.2f} · "
        f"Stop Loss: {entry * (1 - STOP_LOSS_PCT / 100):,.2f}"
    )

# ---- Gráfico de precio + operaciones ----
price_df = df[["time", "close", "sma50"]].rename(columns={"close": "Precio", "sma50": f"SMA{SMA_PERIOD}"})
price_long = price_df.melt("time", var_name="Serie", value_name="valor").dropna()

lines = (
    alt.Chart(price_long)
    .mark_line()
    .encode(
        x=alt.X("time:T", title=None),
        y=alt.Y("valor:Q", scale=alt.Scale(zero=False), title="Precio (USDT)"),
        color=alt.Color("Serie:N", legend=alt.Legend(orient="bottom", title=None)),
    )
)
chart = lines
if state["trades"]:
    tdf = pd.DataFrame(state["trades"])
    tdf["time"] = pd.to_datetime(tdf["time"])
    tdf = tdf[tdf["time"] >= df["time"].min()]
    if not tdf.empty:
        points = (
            alt.Chart(tdf)
            .mark_point(size=150, filled=True, opacity=1)
            .encode(
                x="time:T",
                y="price:Q",
                color=alt.Color(
                    "side:N",
                    scale=alt.Scale(domain=["COMPRA", "VENTA"], range=["#16a34a", "#dc2626"]),
                    legend=alt.Legend(orient="bottom", title="Operación"),
                ),
                shape=alt.Shape("side:N", scale=alt.Scale(domain=["COMPRA", "VENTA"], range=["triangle-up", "triangle-down"])),
                tooltip=["time:T", "side:N", "price:Q", "reason:N"],
            )
        )
        chart = lines + points
st.altair_chart(chart.properties(height=380).interactive(), use_container_width=True)

# ---- RSI ----
with st.expander("RSI (14)", expanded=False):
    rsi_chart = (
        alt.Chart(df.dropna(subset=["rsi"]))
        .mark_line()
        .encode(x=alt.X("time:T", title=None), y=alt.Y("rsi:Q", scale=alt.Scale(domain=[0, 100])))
    )
    rule = alt.Chart(pd.DataFrame({"y": [RSI_OVERSOLD]})).mark_rule(strokeDash=[4, 4], color="orange").encode(y="y:Q")
    st.altair_chart((rsi_chart + rule).properties(height=180), use_container_width=True)
    st.caption(f"RSI actual: {df['rsi'].iloc[-1]:.1f}")

# ---- Historial ----
st.subheader("🧾 Historial de operaciones")
if state["trades"]:
    hist = pd.DataFrame(state["trades"]).iloc[::-1].reset_index(drop=True)
    hist.columns = ["Fecha (UTC)", "Tipo", "Precio", "Cantidad", "Motivo", "P&L (USDT)"]
    st.dataframe(hist, use_container_width=True, hide_index=True)
else:
    st.info("Aún no hay operaciones. El bot espera una señal de compra.")

st.caption("⚠️ Simulación con dinero ficticio. No es asesoría financiera.")

# ---- Auto-refresco ----
if running:
    time.sleep(refresh_s)
    st.rerun()
