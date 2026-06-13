# เป้าหมาย / Mission

**Quant Singularity** = AI Crypto Trader อัตโนมัติบน Binance Futures

- กำไรสูงสุด ขาดทุนน้อยสุด อยู่รอดหลายปี
- เลือกเหรียญ · คิดกลยุทธ์ · เปิด/ปิด · ปรับตัวเอง
- เป้ารายวัน = % ของทุน (`QS_DAILY_TARGET`, default 0.5%/วัน)

## ระบบทำอะไรให้แล้ว (Dual-Engine)

| ชั้น | โมดูล |
|------|--------|
| ข้อมูล | Binance, WebSocket, funding rate |
| สัญญาณ | Coin scanner (สแกนทุก USDT-M futures), 4 strategies (momentum, mean reversion, vwap reversal, funding arb) + passive trend-following, ML (LGBM) |
| AI / Intelligence | AI Brain (Claude), Market Intelligence, Strategy Bandit, Regime Classifier |
| ความเสี่ยง | SL/TP, kill switch, drawdown healing / survival mode, live safety |
| ปรับตัว | Bayesian strategy evolver, adaptive controller, compound manager, profit vault, daily profit engine |
| รายงาน | Daily autopilot summary, `status.py`, `golive_check.py` |
| ทดสอบ | backtest_dual, walkforward, validate_dual (66 checks) |

## คำสั่งสำคัญ

```bash
python main.py                    # รันบอท (paper default)
python scripts/validate_dual.py   # ตรวจระบบทั้งหมด (66 checks)
python scripts/golive_check.py    # เช็คความพร้อมก่อน live
./scripts/status.py               # ดูสถานะปัจจุบัน (equity, drawdown, regime)
```

## เส้นทางสู่เงินจริง

1. Paper 7–14 วัน → readiness ≥ 70
2. Testnet + API
3. Live เล็ก (1–3% ทุน)

**ไม่มีการันตีกำไรทุกวัน** — ต้องพิสูจน์ด้วยสถิติ
