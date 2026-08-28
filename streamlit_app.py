from __future__ import annotations
import io, os
from pathlib import Path
import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from trading_core import Backtester, CsvFeed, NewsBlackout, RiskConfig, StrategyConfig, SignalEngine, ai_commentary, features, resample_m5, session_performance

load_dotenv()
st.set_page_config(page_title="Gold Scalping AI", page_icon="🥇", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding-top:.55rem;padding-bottom:1rem;max-width:100%} h1{font-size:1.55rem!important;margin-bottom:.1rem!important} h2,h3{margin-top:.2rem!important}.stMetric{padding:.1rem 0}.stMetric label{font-size:.78rem}.stMetric [data-testid='stMetricValue']{font-size:1.35rem} div[data-testid='stVerticalBlock']{gap:.45rem}.compact-note{font-size:.78rem;opacity:.78}
</style>""", unsafe_allow_html=True)


def secret(name,default=""):
    try:v=st.secrets.get(name,None)
    except Exception:v=None
    return str(v if v not in (None,"") else os.getenv(name,default))
def get_json(url,timeout=5):
    r=requests.get(url,timeout=timeout); r.raise_for_status(); return r.json()
def bridge_accounts(base): return get_json(base.rstrip("/")+"/accounts")
def load_bridge(base,account,limit=1500):
    r=requests.get(base.rstrip("/")+f"/bars/{account}",params={"limit":limit},timeout=8); r.raise_for_status(); rows=r.json()
    if not rows: raise ValueError("MT5 bridge has no bars yet")
    df=pd.DataFrame(rows); df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.sort_values("timestamp").drop_duplicates("timestamp",keep="last")
    return df[["timestamp","open","high","low","close","spread"]].reset_index(drop=True)
def bridge_zones(base,account):
    try: rows=get_json(base.rstrip("/")+f"/signal_zones/{account}")
    except Exception:return pd.DataFrame()
    if not rows:return pd.DataFrame()
    z=pd.DataFrame(rows); z["timestamp"]=pd.to_datetime(z["timestamp"],utc=True); z["zone_end"]=z["timestamp"]+pd.Timedelta(minutes=15); return z
def save_zone(base,account,sig):
    if sig.side not in ("BUY","SELL"):return
    half=max(float(sig.atr)*.25,.10); payload={"account_id":str(account),"side":sig.side,"timestamp":pd.Timestamp(sig.timestamp).isoformat(),"zone_low":float(sig.entry-half),"zone_high":float(sig.entry+half),"entry":float(sig.entry),"stop":float(sig.stop),"target":float(sig.target),"score":float(sig.score),"reason":str(sig.reason)}
    try: requests.post(base.rstrip("/")+"/signal_zone",json=payload,timeout=3).raise_for_status()
    except Exception:pass
def queue_order(base,account,side,symbol,volume,sl,tp):
    payload={"account_id":str(account),"side":side,"symbol":symbol,"volume":float(volume),"stop_loss":float(sl),"take_profit":float(tp),"note":"Human-confirmed Streamlit order"}
    r=requests.post(base.rstrip("/")+"/manual_order",json=payload,timeout=5); r.raise_for_status(); return r.json()
def confidence(score): return int(max(0,min(100,round(abs(float(score))/2.5*100))))


def gold_chart(frame,zones):
    view=frame.tail(180).copy(); view["direction"]=view.close>=view.open; cols=["low","high","ema9","ema21","ema50"]; lo=float(view[cols].min().min()); hi=float(view[cols].max().max())
    if not zones.empty:
        zv=zones[zones.timestamp>=view.timestamp.min()]
        if not zv.empty: lo=min(lo,float(zv.zone_low.min())); hi=max(hi,float(zv.zone_high.max()))
    span=max(hi-lo,.5); ys=alt.Scale(domain=[lo-span*.06,hi+span*.06],zero=False); x=alt.X("timestamp:T",title=None,axis=alt.Axis(format="%H:%M",labelAngle=0))
    layers=[]
    if not zones.empty:
        zv=zones[zones.zone_end>=view.timestamp.min()].copy()
        if not zv.empty:
            layers.append(alt.Chart(zv).mark_rect(opacity=.12).encode(x="timestamp:T",x2="zone_end:T",y=alt.Y("zone_low:Q",scale=ys,title="GOLD"),y2="zone_high:Q",color=alt.Color("side:N",scale=alt.Scale(domain=["BUY","SELL"],range=["#22c55e","#ef4444"]),legend=None)))
            layers.append(alt.Chart(zv).mark_point(filled=True,size=95).encode(x="timestamp:T",y=alt.Y("entry:Q",scale=ys),shape=alt.Shape("side:N",scale=alt.Scale(domain=["BUY","SELL"],range=["triangle-up","triangle-down"])),color=alt.Color("side:N",scale=alt.Scale(domain=["BUY","SELL"],range=["#22c55e","#ef4444"]),legend=None),tooltip=["side:N",alt.Tooltip("entry:Q",format=".2f"),alt.Tooltip("stop:Q",format=".2f"),alt.Tooltip("target:Q",format=".2f")]))
    layers.append(alt.Chart(view).mark_rule().encode(x=x,y=alt.Y("low:Q",scale=ys,title="GOLD"),y2="high:Q"))
    layers.append(alt.Chart(view).mark_bar(size=4).encode(x=x,y=alt.Y("open:Q",scale=ys),y2="close:Q",color=alt.condition("datum.direction",alt.value("#22c55e"),alt.value("#ef4444"))))
    em=view[["timestamp","ema9","ema21","ema50"]].melt("timestamp",var_name="EMA",value_name="price")
    layers.append(alt.Chart(em).mark_line(strokeWidth=1.35).encode(x=x,y=alt.Y("price:Q",scale=ys),color=alt.Color("EMA:N",legend=alt.Legend(orient="top",direction="horizontal"))))
    chart=layers[0]
    for layer in layers[1:]: chart=chart+layer
    return chart.properties(height=390).interactive(bind_y=False)


bridge_url=secret("MT5_BRIDGE_URL","http://127.0.0.1:8765")
source="FxPro MT5 Bridge"; selected_account=None; selected_meta=None; accounts=[]; labels={}
try:
    accounts=bridge_accounts(bridge_url)
    labels={f"{a['account_mode']} • {a['account_id']} • {a.get('symbol','GOLD')}":a["account_id"] for a in accounts}
except Exception as exc:
    st.error(f"MT5 bridge not reachable: {exc}")

max_spread=.60; vol_ratio=1.80; sl_atr=.90; tp_atr=1.35; be_r=.75; trail_start_r=1.; trail_atr=.60; starting_equity=10000.; risk_pct=.25; max_daily_loss_pct=1.; max_trades=8
with st.sidebar:
    st.header("Connection")
    if labels:
        selected_label=st.selectbox("MT5 account",list(labels.keys()),key="mt5_account_selector")
        selected_account=labels[selected_label]
        selected_meta=next(a for a in accounts if a["account_id"]==selected_account)
        st.success(f"{selected_meta.get('account_mode')} • {selected_account}\n\n{selected_meta.get('symbol','GOLD')} • {selected_meta.get('bars',0)} bars")
    else:
        st.warning("No MT5 account is currently sending data to the bridge.")
    st.caption(bridge_url)
    with st.expander("Strategy settings",expanded=False):
        max_spread=st.number_input("Max spread ($)",.01,10.,.60,.05); vol_ratio=st.number_input("Volatility spike limit",1.,5.,1.80,.05); sl_atr=st.number_input("Stop ATR",.2,5.,.90,.05); tp_atr=st.number_input("Target ATR",.2,8.,1.35,.05)
        be_r=st.number_input("Breakeven trigger (R)",.1,5.,.75,.05); trail_start_r=st.number_input("Trailing starts (R)",.1,5.,1.,.05); trail_atr=st.number_input("Trailing distance ATR",.1,5.,.60,.05)
    with st.expander("Risk / backtest",expanded=False):
        starting_equity=st.number_input("Starting equity",100.,value=10000.,step=500.); risk_pct=st.number_input("Risk / trade %",.01,5.,.25,.05); max_daily_loss_pct=st.number_input("Daily loss kill-switch %",.1,20.,1.,.1); max_trades=st.number_input("Max trades/day",1,100,8,1)

if not selected_account: st.stop()
scfg=StrategyConfig(max_spread=max_spread,vol_spike_ratio=vol_ratio,sl_atr=sl_atr,tp_atr=tp_atr,breakeven_r=be_r,trailing_start_r=trail_start_r,trailing_atr=trail_atr)
rcfg=RiskConfig(starting_equity=starting_equity,risk_per_trade=risk_pct/100,max_daily_loss=max_daily_loss_pct/100,max_trades_per_day=int(max_trades)); news=NewsBlackout.from_csv("sample_news.csv",20,20)
try:
    raw=load_bridge(bridge_url,selected_account); trades,equity,metrics=Backtester(scfg,rcfg,news).run(raw); m1=features(raw,scfg).dropna().reset_index(drop=True); m5=features(resample_m5(raw),scfg).dropna().reset_index(drop=True); latest=SignalEngine(scfg,news).decide_at(m1,m5,len(m1)-1)
except Exception as exc: st.error(str(exc)); st.stop()

st.title("🥇 GOLD Scalping Cockpit")
st.caption(f"{selected_meta.get('account_mode')} • {selected_account} • Human-confirmed execution • Signal engine never clicks BUY/SELL by itself")
left,right=st.columns([3.25,1.15],gap="medium")

with left:
    @st.fragment(run_every="2s")
    def live_panel():
        try:
            live_raw=load_bridge(bridge_url,selected_account); lm1=features(live_raw,scfg).dropna().reset_index(drop=True); lm5=features(resample_m5(live_raw),scfg).dropna().reset_index(drop=True); sig=SignalEngine(scfg,news).decide_at(lm1,lm5,len(lm1)-1); save_zone(bridge_url,selected_account,sig); zones=bridge_zones(bridge_url,selected_account)
        except Exception as exc: st.warning(f"Live refresh: {exc}"); return
        icon="🟢" if sig.side=="BUY" else "🔴" if sig.side=="SELL" else "🟢" if "BUY" in sig.side else "🔴" if "SELL" in sig.side else "🟡"
        a,b,c,d,e=st.columns([1.35,.8,1,1,1]); a.metric("SIGNAL",f"{icon} {sig.side}"); b.metric("Strength",f"{confidence(sig.score)}%"); c.metric("M1 / M5",f"{sig.m1_side}/{sig.m5_side}"); d.metric("Session",sig.session); e.metric("GOLD",f"{lm1.iloc[-1].close:.2f}")
        msg=f"{sig.reason} • {sig.filter_reason}"
        if sig.side in ("BUY","SELL"): st.success(f"{sig.side} • Entry {sig.entry:.2f} • SL {sig.stop:.2f} • TP {sig.target:.2f} • {msg}")
        elif "WATCH" in sig.side: st.warning(f"{sig.side} • Possible zone {sig.entry:.2f} • {msg}")
        else: st.info(msg)
        st.altair_chart(gold_chart(lm1,zones),use_container_width=True)
    live_panel()

with right:
    st.subheader("Manual Trade")
    st.caption("You press the button. AI does not auto-execute.")
    symbol=str(selected_meta.get("symbol","GOLD")); volume=st.number_input("Lots",min_value=.01,value=.01,step=.01,format="%.2f")
    default_sl=float(latest.stop) if latest.stop!=latest.entry else 0.; default_tp=float(latest.target) if latest.target!=latest.entry else 0.
    manual_sl=st.number_input("Stop Loss",min_value=0.,value=max(0.,default_sl),step=.10,format="%.2f"); manual_tp=st.number_input("Take Profit",min_value=0.,value=max(0.,default_tp),step=.10,format="%.2f")
    confirmed=st.checkbox(f"Confirm {selected_meta.get('account_mode')} account")
    b1,b2=st.columns(2)
    if b1.button("🟢 BUY",use_container_width=True,type="primary"):
        if not confirmed: st.error("Confirm account first")
        else:
            try:r=queue_order(bridge_url,selected_account,"BUY",symbol,volume,manual_sl,manual_tp); st.success(f"BUY queued • {r.get('command_id')}")
            except Exception as exc: st.error(str(exc))
    if b2.button("🔴 SELL",use_container_width=True):
        if not confirmed: st.error("Confirm account first")
        else:
            try:r=queue_order(bridge_url,selected_account,"SELL",symbol,volume,manual_sl,manual_tp); st.success(f"SELL queued • {r.get('command_id')}")
            except Exception as exc: st.error(str(exc))
    st.divider(); st.caption("Signal guide"); st.markdown("**BUY / SELL** = M1 + M5 confirmed  \n**WATCH BUY / SELL** = directional setup forming  \n**WAIT** = no useful directional edge")
    st.divider(); st.metric("Backtest PF","∞" if metrics["profit_factor"]==float("inf") else f"{metrics['profit_factor']:.2f}"); st.metric("Win rate",f"{metrics['win_rate_pct']:.1f}%"); st.metric("Max DD",f"{metrics['max_drawdown_pct']:.2f}%")

st.divider()
t1,t2,t3,t4,t5=st.tabs(["🕒 Session performance","🟢🔴 24h zones","🖱 Orders","📒 Backtest","🤖 AI context"])
with t1:
    sp=session_performance(trades)
    if sp.empty: st.info("Session statistics will appear as trades accumulate.")
    else: st.dataframe(sp,use_container_width=True,hide_index=True)
    st.caption("London and New York are context/quality labels, not hard blockers. Current engine can show confirmed setups outside session with an OFF-SESSION warning.")
with t2:
    z=bridge_zones(bridge_url,selected_account)
    if z.empty: st.info("No qualified BUY/SELL zones in the current 24-hour record.")
    else: st.dataframe(z.sort_values("timestamp",ascending=False)[["timestamp","side","zone_low","zone_high","entry","stop","target","score","reason"]],use_container_width=True,hide_index=True)
with t3:
    try:
        h=get_json(bridge_url.rstrip("/")+f"/orders/{selected_account}"); st.dataframe(pd.DataFrame(h.get("commands",[])),use_container_width=True,hide_index=True); st.dataframe(pd.DataFrame(h.get("results",[])),use_container_width=True,hide_index=True)
    except Exception as exc: st.warning(str(exc))
with t4:
    c1,c2,c3,c4,c5=st.columns(5); c1.metric("Trades",metrics["trades"]); c2.metric("Win",f"{metrics['win_rate_pct']:.1f}%"); c3.metric("PF","∞" if metrics["profit_factor"]==float("inf") else f"{metrics['profit_factor']:.2f}"); c4.metric("Avg R",f"{metrics['avg_r']:.2f}"); c5.metric("Net",f"${metrics['net_pnl']:.2f}"); st.dataframe(trades,use_container_width=True,hide_index=True)
with t5:
    if st.button("Generate AI market commentary"):
        text=ai_commentary(m1,latest); st.write(text if text else "OPENAI_API_KEY is not configured. Quant signal engine still runs.")

st.caption("Signals are quantitative/model outputs, not guarantees. Manual execution only. 24h chart zones are short-term visual memory; session performance is evaluated separately.")