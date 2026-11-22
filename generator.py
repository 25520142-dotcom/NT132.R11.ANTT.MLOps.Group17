#!/usr/bin/env python3
"""
Sinh N khách hàng mới (mặc định 500) rồi gửi tới:
  POST {base_url}/predict   -> nhận prediction_id
  POST {base_url}/feedback  -> gửi nhãn thực tế (actual: 0/1) cho prediction_id đó

Ràng buộc dữ liệu:
  * Không trùng với bất kỳ dòng nào trong train_data.csv / test_data.csv
    (so khớp theo toàn bộ 9 feature) và không trùng nhau giữa các khách mới.
  * Mọi giá trị nằm trong khoảng [min, max] (số) / tập giá trị (phân loại) của train_data.
  * Phân phối bám sát train: mỗi khách mới được tạo từ một dòng train ngẫu nhiên
    + nhiễu nhỏ, nên giữ được tương quan giữa các cột.

"actual" được sinh bằng cách huấn luyện một mô hình churn nhỏ trên train+test
rồi lấy mẫu Bernoulli(xác suất churn), để feedback có tính "thực tế" thay vì ngẫu nhiên hoàn toàn.

Cách chạy:
  pip install pandas numpy requests scikit-learn
  python simulate_customers.py --train train_data.csv --test test_data.csv
  python simulate_customers.py --dry-run          # chỉ sinh dữ liệu, không gửi
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import requests

NUM_COLS = ["Age", "Tenure", "Usage Frequency", "Support Calls",
            "Payment Delay", "Total Spend", "Last Interaction"]
INT_COLS = ["Age", "Tenure", "Usage Frequency", "Support Calls",
            "Payment Delay", "Last Interaction"]
CAT_COLS = ["Gender", "Subscription Type", "Contract Length"]
FEATURES = ["Age", "Gender", "Tenure", "Usage Frequency", "Support Calls",
            "Payment Delay", "Subscription Type", "Contract Length",
            "Total Spend", "Last Interaction"]
# Tên cột CSV -> tên field của API
API_NAME = {c: c.replace(" ", "_") for c in FEATURES}


# --------------------------------------------------------------------------- #
# Dữ liệu
# --------------------------------------------------------------------------- #
def load_clean(path):
    df = pd.read_csv(path)
    return df.dropna(subset=FEATURES + ["Churn"]).reset_index(drop=True)


def row_key(age, gender, tenure, usage, calls, delay, sub, contract, spend, last):
    """Khóa so khớp trùng lặp: Total Spend làm tròn 2 chữ số."""
    return (int(age), str(gender), int(tenure), int(usage), int(calls), int(delay),
            str(sub), str(contract), round(float(spend), 2), int(last))


def keys_of(df):
    return set(
        row_key(*r) for r in df[FEATURES].itertuples(index=False, name=None)
    )


def generate_customers(train, existing_keys, n, rng):
    lo = train[NUM_COLS].min()
    hi = train[NUM_COLS].max()
    cats = {c: sorted(train[c].unique()) for c in CAT_COLS}

    # Độ lệch nhiễu: ~5% khoảng giá trị của cột (tối thiểu 1 đơn vị với cột nguyên)
    sigma = {c: max((hi[c] - lo[c]) * 0.05, 1.0) for c in INT_COLS}
    sigma["Total Spend"] = (hi["Total Spend"] - lo["Total Spend"]) * 0.05

    out, seen = [], set(existing_keys)
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError("Không sinh đủ dòng không trùng; hãy tăng nhiễu.")
        base = train.iloc[rng.integers(len(train))]
        rec = {}
        for c in INT_COLS:
            v = int(round(base[c] + rng.normal(0, sigma[c])))
            rec[c] = int(np.clip(v, lo[c], hi[c]))
        spend = float(np.clip(base["Total Spend"] + rng.normal(0, sigma["Total Spend"]),
                              lo["Total Spend"], hi["Total Spend"]))
        rec["Total Spend"] = round(spend, 2)
        for c in CAT_COLS:
            # Giữ đa số giá trị của dòng gốc, thỉnh thoảng đổi (10%) trong tập đã học
            rec[c] = base[c] if rng.random() > 0.10 else rng.choice(cats[c])
        rec["Gender"], rec["Subscription Type"], rec["Contract Length"] = (
            str(rec["Gender"]), str(rec["Subscription Type"]), str(rec["Contract Length"]))

        k = row_key(rec["Age"], rec["Gender"], rec["Tenure"], rec["Usage Frequency"],
                    rec["Support Calls"], rec["Payment Delay"], rec["Subscription Type"],
                    rec["Contract Length"], rec["Total Spend"], rec["Last Interaction"])
        if k in seen:
            continue
        seen.add(k)
        out.append(rec)
    return pd.DataFrame(out)[FEATURES]


def build_label_model(train, test):
    """Mô hình churn để sinh nhãn 'actual' có tính thực tế."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    full = pd.concat([train, test], ignore_index=True)
    X = pd.get_dummies(full[FEATURES], columns=CAT_COLS)
    model = HistGradientBoostingClassifier(max_iter=100, random_state=0)
    model.fit(X, full["Churn"].astype(int))
    return model, list(X.columns)


def sample_actual(model, columns, df, rng):
    X = pd.get_dummies(df[FEATURES], columns=CAT_COLS).reindex(columns=columns, fill_value=0)
    p = model.predict_proba(X)[:, 1]
    return (rng.random(len(p)) < p).astype(int)


# --------------------------------------------------------------------------- #
# Gọi API
# --------------------------------------------------------------------------- #
def to_payload(row):
    p = {API_NAME[c]: row[c] for c in FEATURES}
    for c in INT_COLS:
        p[API_NAME[c]] = int(p[API_NAME[c]])
    p["Total_Spend"] = float(p["Total_Spend"])
    return p


def extract_prediction_id(resp_json):
    """Tìm prediction_id trong response (tên key có thể khác nhau)."""
    if isinstance(resp_json, dict):
        for k in ("prediction_id", "predictionId", "id"):
            if k in resp_json:
                return str(resp_json[k])
        for v in resp_json.values():  # response lồng nhau
            if isinstance(v, dict):
                pid = extract_prediction_id(v)
                if pid:
                    return pid
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_data.csv")
    ap.add_argument("--test", default="test_data.csv")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delay", type=float, default=0.0, help="giây nghỉ giữa các request")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--out", default="new_customers.csv")
    ap.add_argument("--dry-run", action="store_true", help="chỉ sinh dữ liệu, không gọi API")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Đang đọc dữ liệu train/test ...")
    train, test = load_clean(args.train), load_clean(args.test)
    existing = keys_of(train) | keys_of(test)

    print(f"Đang sinh {args.n} khách hàng mới ...")
    customers = generate_customers(train, existing, args.n, rng)

    print("Đang huấn luyện mô hình sinh nhãn feedback ...")
    model, cols = build_label_model(train, test)
    customers["actual"] = sample_actual(model, cols, customers, rng)

    # Kiểm tra lại ràng buộc
    new_keys = keys_of(customers)
    assert len(new_keys) == len(customers), "Có khách mới bị trùng nhau"
    assert not (new_keys & existing), "Có khách mới trùng train/test"
    for c in NUM_COLS:
        assert customers[c].between(train[c].min(), train[c].max()).all(), f"{c} ngoài khoảng"
    for c in CAT_COLS:
        assert customers[c].isin(train[c].unique()).all(), f"{c} ngoài tập giá trị"
    print("OK: không trùng train/test, nằm trong khoảng đã học.")

    if args.dry_run:
        customers.to_csv(args.out, index=False)
        print(f"Dry-run: đã lưu {args.out}")
        return

    session = requests.Session()
    ok_pred = ok_fb = 0
    prediction_ids, errors = [], []
    t0 = time.time()

    for i, row in customers.iterrows():
        payload = to_payload(row)
        pid = None
        try:
            r = session.post(f"{args.base_url}/predict", json=payload, timeout=args.timeout)
            r.raise_for_status()
            pid = extract_prediction_id(r.json())
            if pid is None:
                raise ValueError(f"Response không có prediction_id: {r.text[:200]}")
            ok_pred += 1
        except Exception as e:  # noqa: BLE001
            errors.append((i, "predict", str(e)))

        prediction_ids.append(pid)
        if pid is not None:
            try:
                fb = {"prediction_id": pid, "actual": int(row["actual"])}
                r = session.post(f"{args.base_url}/feedback", json=fb, timeout=args.timeout)
                r.raise_for_status()
                ok_fb += 1
            except Exception as e:  # noqa: BLE001
                errors.append((i, "feedback", str(e)))

        if (i + 1) % 50 == 0 or i + 1 == len(customers):
            print(f"  [{i + 1}/{len(customers)}] predict OK={ok_pred} feedback OK={ok_fb}")
        if args.delay:
            time.sleep(args.delay)

    customers["prediction_id"] = prediction_ids
    customers.to_csv(args.out, index=False)

    print(f"\nHoàn tất sau {time.time() - t0:.1f}s | predict {ok_pred}/{len(customers)} "
          f"| feedback {ok_fb}/{len(customers)} | đã lưu {args.out}")
    if errors:
        print(f"Có {len(errors)} lỗi, ví dụ 5 lỗi đầu:")
        for e in errors[:5]:
            print("  ", e)
        sys.exit(1)


if __name__ == "__main__":
    main()