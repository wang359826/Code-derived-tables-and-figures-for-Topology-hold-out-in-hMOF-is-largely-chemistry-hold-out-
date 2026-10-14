"""空白探测器：比较「实验文献量」与「ML 建模文献量」的比值。

逻辑：某细分方向若实验/实测文献很多、但已有机器学习建模文献很少，
说明该方向的数据已积累但未被人系统挖掘 → 适合做「文献挖掘数据集 + 可解释 ML」。
比值越高 = 越蓝海。同时输出绝对量，避免掉进「没人做的冷门坑」。

仅用标准库，串行限流请求 OpenAlex。
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.openalex.org"
MAILTO = "undergrad-research-planner@example.org"
UA = f"gap-detector/1.0 (mailto:{MAILTO})"
SLEEP = 1.2


def _get(path, params):
    params = dict(params)
    params["mailto"] = MAILTO
    url = f"{BASE}{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                time.sleep(6)
                continue
            print(f"  !! HTTP {exc.code} :: {url}")
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {type(exc).__name__}: {exc}")
            time.sleep(3)
            continue
    return None


def count(search_terms, start_year=2000):
    d = _get(
        "/works",
        {
            "search": search_terms,
            "filter": f"type:article,from_publication_date:{start_year}-01-01",
            "per-page": "1",
        },
    )
    if not d:
        return None
    return d.get("meta", {}).get("count")


# (方向名, 基础实验检索式, 叠加 ML 的检索式)
PAIRS = [
    (
        "沸石/分子筛 · CO2 吸附",
        "zeolite CO2 adsorption capacity",
        "machine learning prediction zeolite CO2 adsorption",
    ),
    (
        "沸石/分子筛 · VOCs 吸附脱除",
        "zeolite adsorption volatile organic compound removal",
        "machine learning prediction zeolite volatile organic compound adsorption",
    ),
    (
        "活性炭 · VOCs 吸附",
        "activated carbon adsorption volatile organic compound",
        "machine learning prediction activated carbon volatile organic compound adsorption",
    ),
    (
        "沸石 · CO2/N2 选择性",
        "zeolite CO2 N2 selectivity adsorption separation",
        "machine learning prediction zeolite CO2 N2 selectivity",
    ),
    (
        "吸附 · 水汽/湿度效应",
        "CO2 adsorption humidity water vapor effect zeolite",
        "machine learning CO2 adsorption humidity water vapor effect",
    ),
    (
        "分子筛膜 · CO2 分离",
        "zeolite membrane CO2 separation permeance",
        "machine learning prediction zeolite membrane CO2 separation",
    ),
    (
        "生物炭 · 抗生素吸附",
        "biochar adsorption antibiotic",
        "machine learning prediction biochar antibiotic adsorption",
    ),
    (
        "MOF · CO2 吸附",
        "metal organic framework CO2 adsorption",
        "machine learning prediction metal organic framework CO2 adsorption",
    ),
    (
        "固废/飞灰合成沸石 · 吸附",
        "zeolite synthesized from fly ash waste adsorption",
        "machine learning zeolite fly ash synthesis adsorption prediction",
    ),
    (
        "吸附剂 · 再生能耗/循环稳定性",
        "adsorbent regeneration energy cyclic stability CO2 capture",
        "machine learning adsorbent regeneration energy cyclic stability prediction",
    ),
]

if __name__ == "__main__":
    print("=" * 88)
    print("空白探测：实验文献量 vs 机器学习建模文献量")
    print("比值越高 → 数据已积累但尚未被系统挖掘 → 越适合做数据集型研究")
    print("=" * 88)
    rows = []
    for name, base_q, ml_q in PAIRS:
        exp_n = count(base_q)
        time.sleep(SLEEP)
        ml_n = count(ml_q, start_year=2015)
        time.sleep(SLEEP)
        if exp_n is None or ml_n is None:
            print(f"{name}: 检索失败")
            continue
        ratio = (exp_n / ml_n) if ml_n else float("inf")
        rows.append((name, exp_n, ml_n, ratio))
        print(f"{name}")
        print(f"    实验/实测文献 = {exp_n:>7}   ML建模文献 = {ml_n:>6}   比值 = {ratio:6.1f}")

    print("-" * 88)
    print("按比值排序（越靠前越蓝海，但需结合绝对量判断可行性）：")
    for name, exp_n, ml_n, ratio in sorted(rows, key=lambda r: -r[3]):
        flag = ""
        if exp_n < 1500:
            flag = "  ⚠ 文献池偏小，数据可能不够"
        elif exp_n > 25000:
            flag = "  ⚠ 过大，筛选成本高"
        print(f"   {ratio:6.1f}  | 实验 {exp_n:>7} / ML {ml_n:>6} | {name}{flag}")

    with open("gap_detector_results.json", "w", encoding="utf-8") as fh:
        json.dump(
            [{"direction": n, "experimental": e, "ml": m, "ratio": r} for n, e, m, r in rows],
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print("-" * 88)
    print("已保存: gap_detector_results.json")
