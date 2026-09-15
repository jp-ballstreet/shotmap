"""pl_shotmap_app.py v3 — where shots come from, relative to average.

v3 (2026-09-12)
  SEASONS   every understat_*_all*.parquet in data/processed is loaded
            (unsuffixed = 25-26). Pick the season to view and the BASELINE
            the "vs average" compares against: the same season to date, or
            the full prior season (stable reference early in a season).
  KEYS      player_id everywhere (name collisions: Vitinha, Wesley, ...).
  ROLE      shooter position from Understat codes, minutes-weighted, 'S'
            (sub) stripped — the old mapping sent 329 sub-only players to GK.
  ARCHETYPE from 48_player_archetypes_v2.py (role x shooting style, all
            five leagues, multi-season pooled); falls back to rule labels.
  FINDER    tab: pick a defence, see its schedule-adjusted archetype bleed,
            then list an attacking team's (or the league's) players in
            those archetypes with season lines and a fit ranking.
  TEAM      heat map shrunk toward the baseline for thin samples
            (n/(n+6) matches) with an n-matches badge; archetype and zone
            concession tables are SCHEDULE-ADJUSTED: expected = what the
            opponents actually faced generate against everyone else.

Reads: data/processed/understat_{shots,team,players}_all*.parquet
       data/derived/player_archetypes_v2.csv (optional)
Run:   streamlit run pl_shotmap_app.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
DERIVED = ROOT / "data" / "derived"
PAPER, INK, RUST, MUTED, STEEL = "#F3EFE4", "#23201B", "#C0432E", "#8A8378", "#3E5C76"

st.set_page_config(page_title="shot maps", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp {{
  background-color:{PAPER};
  background-image: linear-gradient(rgba(35,32,27,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(35,32,27,.04) 1px, transparent 1px);
  background-size:24px 24px; color:{INK};
  font-family:'JetBrains Mono',monospace; }}
h1,h2,h3 {{ text-transform:lowercase; }}
[data-testid="stHeader"] {{ background:transparent; }}
.badge {{ display:inline-block; padding:2px 8px; border:1px solid {INK};
  border-radius:3px; font-size:0.8rem; margin-right:6px; }}
.warn {{ color:{RUST}; }}
</style>""", unsafe_allow_html=True)

NX, NY, X0, SIG = 30, 20, 0.5, 1.1
SHRINK_K = 6                      # matches of prior weight for team grids


def zone(x, y):
    if x >= 0.9476 and 0.365 <= y <= 0.635:
        return "six-yard"
    if x >= 0.843 and 0.365 <= y <= 0.635:
        return "box central"
    if x >= 0.843 and 0.2035 <= y <= 0.7965:
        return "box wide"
    if x >= 0.75 and 0.365 <= y <= 0.635:
        return "zone 14"
    return "deep/wide"


def role_from_position(p) -> str:
    s = str(p)
    if "GK" in s:
        return "GK"
    s = s.replace("S", "")
    return "FW" if "F" in s else "MF" if "M" in s else "DF" if "D" in s else "UNK"


ARCH_RULES = [
    ("poacher", lambda c: c["six-yard"] + c["box central"] >= 0.62 and c["hdr"] < 0.30),
    ("aerial threat", lambda c: c["hdr"] >= 0.30),
    ("left-channel forward", lambda c: c["side_l"] >= 0.45 and c["box"] >= 0.5),
    ("right-channel forward", lambda c: c["side_r"] >= 0.45 and c["box"] >= 0.5),
    ("edge-of-box shooter", lambda c: c["zone 14"] + c["deep/wide"] >= 0.45),
    ("mixed attacker", lambda c: True),
]


def _tag(path: Path) -> str:
    m = re.search(r"_all_(\d{4})\.parquet$", path.name)
    return m.group(1) if m else "2526"


def _season_label(tag: str) -> str:
    return f"20{tag[:2]}-{tag[2:]}"


@st.cache_data(show_spinner=False)
def load():
    def read_all(kind):
        parts = []
        for f in sorted(PROCESSED.glob(f"understat_{kind}_all*.parquet")):
            d = pd.read_parquet(f)
            d["season_tag"] = _tag(f)
            parts.append(d)
        if not parts:
            raise FileNotFoundError(f"understat_{kind}_all*.parquet")
        return pd.concat(parts, ignore_index=True)
    sh = read_all("shots").drop_duplicates(["season_tag", "shot_id"])
    ut = read_all("team").drop_duplicates(["season_tag", "game_id", "team"])
    up = read_all("players").drop_duplicates(["season_tag", "player_id", "team_raw"])

    # canonical shooting team + conceding team, within season
    bridge = ut[["season_tag", "game_id", "team_raw", "team"]].drop_duplicates()
    sh = sh.merge(bridge, left_on=["season_tag", "game_id", "team"],
                  right_on=["season_tag", "game_id", "team_raw"],
                  how="left", suffixes=("", "_c"))
    sh["team_c"] = sh.team_c.astype(object).fillna(sh.team.astype(object)).astype(str)
    pair = ut[["season_tag", "game_id", "team"]].drop_duplicates()
    opp = pair.merge(pair, on=["season_tag", "game_id"])
    opp = opp[opp.team_x != opp.team_y].rename(
        columns={"team_x": "team_c", "team_y": "conceding"})
    sh = sh.merge(opp, on=["season_tag", "game_id", "team_c"], how="left")
    sh = sh[sh.result != "Own Goal"].copy()
    for c in ("player", "team", "conceding", "league", "situation", "result"):
        sh[c] = sh[c].astype(object)                # plain python strings: no
    # arrow-str vs object clashes in string concatenation (pandas 3)
    sh["is_goal"] = (sh.result == "Goal").astype(float)
    sh["one"] = 1.0
    sh["d"] = pd.to_datetime(sh.date)
    sh["x"] = sh.location_x.clip(X0 + 1e-6, 1 - 1e-6)
    sh["y"] = sh.location_y.clip(1e-6, 1 - 1e-6)
    sh["zone"] = [zone(x, y) for x, y in zip(sh.x, sh.y)]
    sh["shot_type"] = np.where(sh.body_part.isna(), "header/other", "footed")
    sh["situation"] = sh.situation.fillna("Penalty")
    sh["side"] = pd.cut(sh.y, [0, 0.365, 0.635, 1.0],
                        labels=["right", "central", "left"])

    # role: most-minutes player-season row, S stripped
    up["role"] = up.position.map(role_from_position)
    known = up[up.role != "UNK"].sort_values("minutes", ascending=False)
    roles = known.drop_duplicates("player_id")[["player_id", "role"]]

    # archetypes: v2 (player_id, all leagues) else in-app rule labels
    v2 = DERIVED / "player_archetypes_v2.csv"
    if v2.exists():
        fa = pd.read_csv(v2)[["player_id", "role", "archetype", "confidence"]]
        arch = fa.rename(columns={"role": "role_v2"})
        arch_src = f"48 v2 ({len(fa)} players)"
    else:
        feats = []
        for pid, g in sh.groupby("player_id"):
            if len(g) < 20:
                continue
            zs = g.zone.value_counts(normalize=True)
            c = {z: zs.get(z, 0.0) for z in
                 ("six-yard", "box central", "box wide", "zone 14", "deep/wide")}
            c["box"] = c["six-yard"] + c["box central"] + c["box wide"]
            c["hdr"] = (g.shot_type == "header/other").mean()
            c["side_l"] = (g.side == "left").mean()
            c["side_r"] = (g.side == "right").mean()
            feats.append((pid, next(n for n, r in ARCH_RULES if r(c)), np.nan))
        arch = pd.DataFrame(feats, columns=["player_id", "archetype", "confidence"])
        arch["role_v2"] = np.nan
        arch_src = "in-app rule labels (run 48 for v2)"
    sh = sh.merge(roles, on="player_id", how="left") \
           .merge(arch, on="player_id", how="left")
    sh["pos"] = sh.role.fillna(sh.role_v2).fillna("UNK")
    sh["archetype"] = sh.archetype.astype(object).fillna("low-volume " + sh.pos.astype(str))
    sh = sh.drop(columns=["role", "role_v2"])
    mins = up.groupby(["season_tag", "player_id"]).minutes.sum().reset_index()
    return sh, ut, arch_src, mins


def grid(df, w):
    h, _, _ = np.histogram2d(df.x, df.y, bins=[NX, NY],
                             range=[[X0, 1], [0, 1]], weights=df[w])
    return gaussian_filter(h, SIG)


def league_grid(df, w):
    """Mean of per-team per-match conceded grids."""
    nm = df.groupby("conceding").game_id.nunique()
    gs = [grid(g, w) / max(float(nm.get(t, 1)), 1.0)
          for t, g in df.groupby("conceding")]
    return np.mean(gs, axis=0) if gs else np.zeros((NX, NY))


def pitch_shapes():
    L = dict(color=INK, width=1.4)
    return [dict(type="rect", x0=X0, x1=1, y0=0, y1=1, line=L),
            dict(type="rect", x0=0.843, x1=1, y0=0.2035, y1=0.7965, line=L),
            dict(type="rect", x0=0.9476, x1=1, y0=0.365, y1=0.635, line=L),
            dict(type="line", x0=1, x1=1, y0=0.44, y1=0.56,
                 line=dict(color=INK, width=5)),
            dict(type="circle", x0=0.883, x1=0.893, y0=0.495, y1=0.505,
                 line=L, fillcolor=INK)]


def heat_fig(diff, goals, title, marker_label, hover_who, metric):
    vmax = np.abs(diff).max() or 1.0
    xs = np.linspace(X0, 1, NX + 1)[:-1] + (1 - X0) / NX / 2
    ys = np.linspace(0, 1, NY + 1)[:-1] + 1 / NY / 2
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=diff.T, x=xs, y=ys, zmin=-vmax, zmax=vmax,
        colorscale=[[0, STEEL], [0.5, PAPER], [1, RUST]],
        colorbar=dict(title=f"{metric} vs avg", tickfont=dict(color=INK), len=0.8),
        hovertemplate="%{z:+.2f} " + metric + " vs average<extra></extra>"))
    if len(goals):
        fig.add_trace(go.Scatter(
            x=goals.x, y=goals.y, mode="markers",
            marker=dict(symbol="circle-open-dot", size=13, color=INK,
                        line=dict(width=2.4)),
            name=marker_label,
            text=[f"{who}<br>{d:%d %b} min {m:.0f} — xG {g:.2f}"
                  for who, d, m, g in zip(hover_who, goals.d, goals.minute, goals.xg)],
            hovertemplate="%{text}<extra></extra>"))
    for s in pitch_shapes():
        fig.add_shape(**s)
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family="JetBrains Mono", color=INK, size=11),
        title=dict(text=title, font=dict(size=12)),
        xaxis=dict(visible=False, range=[X0 - 0.01, 1.03]),
        yaxis=dict(visible=False, range=[-0.03, 1.03], scaleanchor="x"),
        height=430, legend=dict(bgcolor=PAPER, orientation="h", y=-0.06),
        margin=dict(t=44, b=6, l=6, r=6))
    return fig


def per_match(df, dim, w):
    """team x level -> per-match totals, and per-team match counts."""
    nm = df.groupby("conceding").game_id.nunique()
    tot = df.groupby(["conceding", dim], observed=True)[w].sum().unstack(fill_value=0)
    return tot.div(nm, axis=0), nm


def schedule_adjusted(df, team, dim, w, base_df):
    """Actual per-match conceded by `dim` vs (a) baseline league average and
    (b) schedule expectation: for each opponent faced, what that opponent's
    shooters of each level generate per match against everyone EXCEPT
    `team`, averaged over the matches played. Percentile of the delta
    across the league's defences (100 = bleeds most)."""
    nm_all = df.groupby("conceding").game_id.nunique()
    gen_tot = df.groupby(["team_c", dim], observed=True)[w].sum().unstack(fill_value=0)
    gm_all = df.groupby("team_c").game_id.nunique()
    # per-team expectation
    def expect_for(t):
        mine = df[df.conceding == t]
        opps = mine.groupby("team_c").game_id.nunique()          # matches vs each opp
        vs_me = mine.groupby(["team_c", dim], observed=True)[w].sum() \
                    .unstack(fill_value=0).reindex(opps.index, fill_value=0)
        rate = (gen_tot.reindex(opps.index, fill_value=0) - vs_me) \
            .div((gm_all.reindex(opps.index) - opps).clip(lower=1), axis=0)
        return rate.mul(opps, axis=0).sum() / max(float(opps.sum()), 1.0)
    act_all, _ = per_match(df, dim, w)
    exp_all = pd.DataFrame({t: expect_for(t) for t in act_all.index}).T \
        .reindex(columns=act_all.columns, fill_value=0)
    delta_all = act_all - exp_all
    pct = delta_all.rank(pct=True).mul(100).round(0)
    base_pm, _ = per_match(base_df, dim, w)
    out = pd.DataFrame({
        "per match": act_all.loc[team],
        "league avg": base_pm.mean().reindex(act_all.columns).fillna(0),
        "schedule exp.": exp_all.loc[team],
    })
    out["vs schedule"] = out["per match"] - out["schedule exp."]
    out["pctile"] = pct.loc[team].astype(int)
    out = out[(out["per match"] > 0) | (out["league avg"] >= 0.005)]
    return out.round(2).sort_values("vs schedule", ascending=False)


def delta_table(mine, league, dim, w, n_mine, n_league):
    a = mine.groupby(dim, observed=True)[w].sum() / n_mine
    b = league.groupby(dim, observed=True)[w].sum() / n_league
    out = pd.DataFrame({"per match": a, "league avg": b}).fillna(0)
    out["vs avg"] = out["per match"] - out["league avg"]
    return out.round(2).sort_values("vs avg", ascending=False)


try:
    SH, UT, ARCH_SRC, MINS = load()
except FileNotFoundError as e:
    st.error(f"missing input: {e} — run 37_pull_understat_all.py")
    st.stop()

st.title("shot maps")
SEASONS = sorted(SH.season_tag.unique(), reverse=True)
c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.4, 2])
season = c1.selectbox("season", SEASONS, format_func=_season_label, key="season")
prior = [s for s in SEASONS if s < season]
base_opts = ["same season (to date)"] + \
    [f"prior season ({_season_label(prior[0])}, full)"] if prior else ["same season (to date)"]
baseline = c2.radio("'vs average' baseline", base_opts, key="baseline")
metric = c3.radio("metric (both panels)", ["shots", "xG", "goals"],
                  horizontal=True, key="metric")
w = {"shots": "one", "xG": "xg", "goals": "is_goal"}[metric]

sh_season = SH[SH.season_tag == season]
dmin, dmax = sh_season.d.min().date(), sh_season.d.max().date()
d_lo, d_hi = c4.slider("period", min_value=dmin, max_value=dmax,
                       value=(dmin, dmax), format="DD MMM YY", key="period")
sh = sh_season[(sh_season.d.dt.date >= d_lo) & (sh_season.d.dt.date <= d_hi)]
if len(sh) == 0:
    st.warning("no shots in the selected period.")
    st.stop()
base_all = sh if baseline.startswith("same") else SH[SH.season_tag == prior[0]]
min_shots = st.sidebar.slider("min shots (player list)", 5, 80, 15, 5, key="min_shots")
shrink = st.sidebar.checkbox("shrink thin-sample team maps toward baseline",
                             value=True, key="shrink")
st.sidebar.caption(f"archetypes: {ARCH_SRC}")
LEAGUES = sorted(sh.league.dropna().unique())
tab_maps, tab_find = st.tabs(["shot maps", "matchup finder"])
left, right = tab_maps.columns(2, gap="large")

# ================= TEAM PANEL =================
with left:
    st.subheader("team — shots allowed")
    lg_t = st.selectbox("league", LEAGUES,
                        index=LEAGUES.index("EPL") if "EPL" in LEAGUES else 0,
                        key="team_lg")
    sh_t = sh[sh.league == lg_t]
    base_t = base_all[base_all.league == lg_t]
    teams = sorted(sh_t.conceding.dropna().unique())
    team = st.selectbox("defence", teams, key="team_sel")
    mine = sh_t[sh_t.conceding == team]
    wm = sh_t.groupby("conceding").game_id.nunique()
    nm = max(float(wm.get(team, 1)), 1.0)
    g_team = grid(mine, w) / nm
    g_base = league_grid(base_t, w)
    if shrink:
        g_team = (nm * g_team + SHRINK_K * g_base) / (nm + SHRINK_K)
    thin = nm < 8
    st.markdown(
        f"<span class='badge'>{int(nm)} matches</span>"
        f"<span class='badge'>baseline: {baseline}</span>"
        + (f"<span class='badge warn'>thin sample — "
           f"{'shrunk' if shrink else 'unshrunk'}</span>" if thin else ""),
        unsafe_allow_html=True)
    goals5 = mine[mine.is_goal == 1].sort_values("d").tail(5)
    st.plotly_chart(heat_fig(
        g_team - g_base, goals5,
        f"{team.lower()} — {metric} allowed vs average, per match",
        "last 5 goals conceded",
        goals5.player.astype(str) + " (" + goals5.team_c.astype(str) + ")", metric),
        use_container_width=True, key="team_map")

    n_base = float(base_t.groupby("conceding").game_id.nunique().sum())
    st.markdown(f"**how the {metric} arrive** (per match, vs {lg_t} baseline)")
    t1, t2 = st.columns(2)
    t1.dataframe(delta_table(mine, base_t, "shot_type", w, nm, n_base),
                 use_container_width=True)
    t2.dataframe(delta_table(mine, base_t, "situation", w, nm, n_base),
                 use_container_width=True)
    t3, t4 = st.columns(2)
    t3.dataframe(delta_table(mine, base_t, "pos", w, nm, n_base),
                 use_container_width=True)
    t4.dataframe(delta_table(mine, base_t, "side", w, nm, n_base),
                 use_container_width=True)
    st.markdown(f"**{metric} allowed by shooter archetype** — schedule-adjusted")
    st.dataframe(schedule_adjusted(sh_t, team, "archetype", w, base_t),
                 use_container_width=True)
    st.markdown(f"**{metric} allowed by zone** — schedule-adjusted")
    st.dataframe(schedule_adjusted(sh_t, team, "zone", w, base_t),
                 use_container_width=True)
    st.caption("schedule exp. = what the opponents this defence actually "
               "faced generate, per match, against every OTHER defence; "
               "'vs schedule' > 0 means they concede more of this than those "
               "attackers normally produce. pctile 100 = bleeds most in league.")

# ================= PLAYER PANEL =================
with right:
    st.subheader("player — shots taken")
    lg_p = st.selectbox("league", LEAGUES,
                        index=LEAGUES.index("EPL") if "EPL" in LEAGUES else 0,
                        key="player_lg")
    sh_p = sh[sh.league == lg_p]
    base_p = base_all[base_all.league == lg_p]
    counts = sh_p.groupby("player_id").agg(n=("one", "sum"), name=("player", "first"),
                                           team=("team_c", "last"))
    counts = counts[counts.n >= min_shots].sort_values("n", ascending=False)
    labels = {pid: f"{r['name']} ({r.team}, {int(r.n)})" for pid, r in counts.iterrows()}
    pid = st.selectbox(f"player (≥{min_shots} shots; {len(counts)})",
                       list(labels), format_func=labels.get, key="player_sel")
    pmine = sh_p[sh_p.player_id == pid]
    player = pmine.player.iloc[0]
    conf = pmine.confidence.iloc[0]
    conf_txt = f" (conf {conf:.2f})" if pd.notna(conf) else ""
    st.caption(f"archetype: **{pmine.archetype.iloc[0]}**{conf_txt} · "
               f"{pmine.pos.iloc[0]} · {int(len(pmine))} shots · "
               f"xG/shot {pmine.xg.mean():.3f} · "
               f"goals−xG {pmine.is_goal.sum() - pmine.xg.sum():+.1f}")
    gp = grid(pmine, w)
    ga = grid(base_p, w)
    diff_p = gp - ga * (gp.sum() / ga.sum() if ga.sum() else 0)
    pg5 = pmine[pmine.is_goal == 1].sort_values("d").tail(5)
    st.plotly_chart(heat_fig(
        diff_p, pg5,
        f"{player.lower()} — {metric} profile vs league shooter of equal volume",
        "last 5 goals scored", "vs " + pg5.conceding.astype(str).fillna("?"), metric),
        use_container_width=True, key="player_map")

    st.markdown(f"**his {metric}, as shares** (vs the league shooter)")

    def share_table(dim):
        a = pmine.groupby(dim, observed=True)[w].sum() / max(pmine[w].sum(), 1e-9)
        b = base_p.groupby(dim, observed=True)[w].sum() / max(base_p[w].sum(), 1e-9)
        out = pd.DataFrame({"his share": a, "league": b}).fillna(0)
        out["vs avg"] = out["his share"] - out["league"]
        return ((out * 100).round(0).astype(int).astype(str) + "%")

    p1, p2 = st.columns(2)
    p1.dataframe(share_table("shot_type"), use_container_width=True)
    p2.dataframe(share_table("situation"), use_container_width=True)
    p3, p4 = st.columns(2)
    p3.dataframe(share_table("zone"), use_container_width=True)
    p4.dataframe(share_table("side"), use_container_width=True)

    same = SH[(SH.player_id == pid) & (SH.season_tag != season)]
    if len(same):
        st.markdown("**other seasons on file**")
        st.dataframe(same.groupby("season_tag").agg(
            shots=("one", "sum"), xG=("xg", "sum"), goals=("is_goal", "sum"),
            xg_per_shot=("xg", "mean")).round(2).rename(index=_season_label),
            use_container_width=True)

# ================= MATCHUP FINDER =================
def player_table(df, mins_season, arch_filter=None, min_n=1):
    """Per-player season line: team, archetype, shots, xG, goals, xG/shot,
    xG/90 (Understat minutes), shots/90, confidence."""
    g = df.groupby("player_id").agg(
        player=("player", "first"), team=("team_c", "last"),
        league=("league", "first"), pos=("pos", "first"),
        archetype=("archetype", "first"), conf=("confidence", "first"),
        shots=("one", "sum"), xG=("xg", "sum"), goals=("is_goal", "sum"),
        setp_share=("situation", lambda x: x.isin(["From Corner", "Set Piece"]).mean()))
    g = g.merge(mins_season.set_index("player_id").minutes, left_index=True,
                right_index=True, how="left")
    g["xG/shot"] = g.xG / g.shots
    g["xG/90"] = g.xG / g.minutes.replace(0, np.nan) * 90
    g["shots/90"] = g.shots / g.minutes.replace(0, np.nan) * 90
    g["G-xG"] = g.goals - g.xG
    if arch_filter:
        g = g[g.archetype.isin(arch_filter)]
    g = g[g.shots >= min_n]
    cols = ["player", "team", "league", "pos", "archetype", "shots", "xG", "goals",
            "G-xG", "xG/shot", "xG/90", "shots/90", "minutes", "setp_share", "conf"]
    return g[cols].round(2).sort_values("xG", ascending=False)


with tab_find:
    st.subheader("matchup finder — who fits the weakness")
    st.caption("pick the DEFENCE; its schedule-adjusted archetype bleed comes "
               "up on the left. then either pick the attacking team to see "
               "which of its players carry those archetypes, or search any "
               "archetype league-wide. player lines are the selected season "
               "and period; archetype labels are pooled (48).")
    f1, f2, f3 = st.columns([1.2, 1.4, 1.4])
    lg_f = f1.selectbox("league", LEAGUES,
                        index=LEAGUES.index("EPL") if "EPL" in LEAGUES else 0,
                        key="find_lg")
    sh_f = sh[sh.league == lg_f]
    base_f = base_all[base_all.league == lg_f]
    teams_f = sorted(sh_f.conceding.dropna().unique())
    defence = f2.selectbox("defence (opponent)", teams_f, key="find_def")
    attack_opts = ["— any team (search by archetype) —"] + \
        [t for t in teams_f if t != defence]
    attack = f3.selectbox("attacking team", attack_opts, key="find_att")
    mins_season = MINS[MINS.season_tag == season]

    fl, fr = st.columns([1, 1.6], gap="large")
    with fl:
        st.markdown(f"**{defence.lower()} — {metric} bled by archetype** "
                    "(schedule-adjusted)")
        weak = schedule_adjusted(sh_f, defence, "archetype", w, base_f)
        weak = weak[~weak.index.str.startswith(("low-volume", "goalkeeper"))]
        st.dataframe(weak, use_container_width=True)
        top_weak = weak[weak["vs schedule"] > 0].index.tolist()
        st.markdown(f"**{defence.lower()} — {metric} bled by zone**")
        st.dataframe(schedule_adjusted(sh_f, defence, "zone", w, base_f),
                     use_container_width=True)
    with fr:
        arch_all = sorted(a for a in sh.archetype.unique()
                          if not a.startswith(("low-volume", "goalkeeper")))
        default_arch = top_weak[:3] if top_weak else []
        picks = st.multiselect("archetypes (defaults to the top bleeds)",
                               arch_all, default=default_arch, key="find_arch")
        min_n_f = st.slider("min shots this season", 1, 30, 3, key="find_min")
        if attack.startswith("—"):
            pool = sh_f[sh_f.team_c != defence]          # not the defence's own men
            head = f"{lg_f} shooters in {', '.join(picks) if picks else 'any archetype'}"
        else:
            pool = sh_f[sh_f.team_c == attack]
            head = f"{attack.lower()} shooters vs {defence.lower()}"
        tbl = player_table(pool, mins_season, picks or None, min_n_f)
        # rank by the defence's bleed to the player's archetype, then by xG
        bleed = weak["vs schedule"]
        tbl["def bleed"] = tbl.archetype.map(bleed).fillna(0).round(2)
        tbl["fit"] = (tbl["def bleed"].clip(lower=0) * tbl["xG/90"].fillna(0)).round(3)
        tbl = tbl.sort_values(["fit", "xG"], ascending=False)
        front = ["player", "team", "archetype", "fit", "def bleed"]
        tbl = tbl[front + [c for c in tbl.columns if c not in front]]
        st.markdown(f"**{head}** — {len(tbl)} players")
        st.dataframe(tbl.reset_index(drop=True), use_container_width=True,
                     height=min(60 + 35 * len(tbl), 720))
        st.caption("def bleed = the defence's 'vs schedule' delta for the "
                   "player's archetype (per match). fit = max(def bleed, 0) x "
                   "player xG/90 — a ranking key, not a projection. players "
                   "with no season minutes on file show NaN per-90s.")

st.caption(f"{_season_label(season)} · period {d_lo:%d %b %y} – {d_hi:%d %b %y} "
           f"· baseline {baseline}. red = more than average from that spot, "
           "blue = less; grids smoothed; own goals excluded; goal on the right. "
           "'header/other' is inferred from the pull's body_part gap. side is "
           "from the shooter's perspective. archetypes are full-profile labels "
           "(48 v2: role x shooting style, multi-season pooled) and do not "
           "change with the period.")
