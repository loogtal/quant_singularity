# เป้าหมาย / Mission

**Quant Singularity** = AI Crypto Trader อัตโนมัติบน Binance Futures

- กำไรสูงสุด ขาดทุนน้อยสุด อยู่รอดหลายปี
- เลือกเหรียญ · คิดกลยุทธ์ · เปิด/ปิด · ปรับตัวเอง
- เป้ารายวัน = % ของทุน (`QS_DAILY_TARGET`, default 0.5%/วัน)

## ระบบทำอะไรให้แล้ว

| ชั้น | โมดูล |
|------|--------|
| ข้อมูล | Binance, WebSocket, funding |
| สัญญาณ | Scanner, Alpha, ML, 4 strategies |
| ความเสี่ยง | SL/TP, kill switch, live safety |
| ปรับตัว | Auto-tuner, reflection, online ML |
| รายงาน | Daily report, live readiness score |
| ทดสอบ | backtest, walkforward, validate |

## คำสั่งสำคัญ

```bash
python main.py                    # รันบอท (paper default)
python scripts/validate.py        # ตรวจระบบทั้งหมด
python scripts/daily_report.py    # รายงานรายวัน
python scripts/preflight.py       # ก่อน live
```

## เส้นทางสู่เงินจริง

1. Paper 7–14 วัน → readiness ≥ 70
2. Testnet + API
3. Live เล็ก (1–3% ทุน)

**ไม่มีการันตีกำไรทุกวัน** — ต้องพิสูจน์ด้วยสถิติ
