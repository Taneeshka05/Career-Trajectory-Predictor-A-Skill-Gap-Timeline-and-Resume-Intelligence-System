import numpy as np
import plotly.express as px
import plotly.graph_objects as go

INDIGO = "#4f46e5"


def gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), title={"text": title},
        number={"suffix": "%" if "Score" not in title else ""},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": INDIGO},
               "steps": [{"range": [0, 40], "color": "#fee2e2"},
                         {"range": [40, 70], "color": "#fef3c7"},
                         {"range": [70, 100], "color": "#dcfce7"}]}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def skill_gap_bar(missing, weeks_by_skill):
    fig = go.Figure(go.Bar(x=[weeks_by_skill[s] for s in missing], y=missing, orientation="h",
                           marker_color="#ef4444"))
    fig.update_layout(title="Missing skills (base learning weeks)", height=max(250, 40 * len(missing) + 100),
                      xaxis_title="Weeks", yaxis=dict(autorange="reversed"),
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


def gantt(df):
    fig = go.Figure()
    for r in df.itertuples():
        color = "#fb923c" if r.type == "foundation" else px.colors.sequential.Purples[min(r.difficulty + 2, 8)]
        fig.add_trace(go.Bar(y=[r.skill], x=[r.weeks], base=[r.start], orientation="h", marker_color=color,
                             hovertemplate=f"{r.skill} ({r.type})<br>week {r.start:.1f} to {r.end:.1f}<extra></extra>",
                             showlegend=False))
    fig.update_layout(title="Learning roadmap (orange = foundation skill, purple = target skill)",
                      xaxis_title="Week", yaxis=dict(autorange="reversed"),
                      height=max(260, 45 * len(df) + 110), margin=dict(l=20, r=20, t=50, b=20))
    return fig


def radar(coverage):
    names = list(coverage.keys())
    values = [coverage[n] * 100 for n in names]
    fig = go.Figure(go.Scatterpolar(r=values + values[:1], theta=names + names[:1], fill="toself",
                                    line_color=INDIGO))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), showlegend=False, height=380,
                      title="Skill coverage by career domain (%)", margin=dict(l=40, r=40, t=60, b=20))
    return fig


def career_map(map_df, point):
    sample = map_df.sample(1500, random_state=1).copy()
    rng = np.random.default_rng(3)
    sample["x"] += rng.normal(0, 0.03, len(sample))
    sample["y"] += rng.normal(0, 0.03, len(sample))
    fig = px.scatter(sample, x="x", y="y", color="domain", opacity=0.45,
                     labels={"x": "Principal component 1", "y": "Principal component 2"})
    fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode="markers+text", text=["You"],
                             textposition="top center", name="You",
                             marker=dict(size=20, symbol="star", color="black")))
    fig.update_layout(height=520, title="Skill space (PCA of 5,500 profiles)")
    return fig


def confusion(cm, labels, title):
    fig = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"))
    fig.update_layout(title=title, height=380)
    return fig


def curves(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history["loss"], name="train loss"))
    fig.add_trace(go.Scatter(y=history["val_loss"], name="validation loss"))
    fig.add_trace(go.Scatter(y=history["accuracy"], name="train accuracy", yaxis="y2", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(y=history["val_accuracy"], name="validation accuracy", yaxis="y2",
                             line=dict(dash="dot")))
    fig.update_layout(title="Neural network training", xaxis_title="Epoch", height=380,
                      yaxis=dict(title="Loss"), yaxis2=dict(title="Accuracy", overlaying="y", side="right"))
    return fig


def metric_bars(df, metric, title):
    fig = px.bar(df, x="Model", y=metric, text=df[metric].round(3), color="Model")
    fig.update_layout(title=title, showlegend=False, height=350)
    return fig


def reg_scatter(df):
    fig = px.scatter(df, x="actual", y="predicted", opacity=0.5,
                     labels={"actual": "Actual weeks", "predicted": "Predicted weeks"})
    top = float(max(df["actual"].max(), df["predicted"].max()))
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color="red"), name="ideal"))
    fig.update_layout(title="Time-to-ready: actual vs predicted", height=380)
    return fig


def rank_bar(df):
    fig = px.bar(df, x="fit_score", y="title", orientation="h", color="skill_match_pct",
                 color_continuous_scale="Viridis", hover_data=["level", "weeks_to_ready"])
    fig.update_layout(title="Best matching jobs", height=max(300, 38 * len(df) + 120),
                      yaxis=dict(autorange="reversed"))
    return fig


def trend(df):
    fig = px.line(df, x="created_at", y=["match_pct", "strength"], markers=True,
                  labels={"value": "Score", "created_at": "Date", "variable": ""})
    fig.update_layout(title="Your progress over time", height=340)
    return fig

LEVEL_COLORS = {"Junior": "#ef4444", "Mid": "#f59e0b", "Senior": "#16a34a"}


def trajectory_chart(traj):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=traj["week"], y=traj["skill_match_pct"], mode="lines", name="Skill match %",
                             line=dict(color=INDIGO)))
    fig.add_trace(go.Scatter(x=traj["week"], y=traj["match_pct"], mode="lines", name="Text match %",
                             line=dict(color="#94a3b8", dash="dot")))
    fig.add_trace(go.Scatter(x=traj["week"], y=traj["skill_match_pct"], mode="markers+text",
                             text=traj["skill"], textposition="top center", showlegend=False,
                             marker=dict(size=13, color=[LEVEL_COLORS[l] for l in traj["level"]],
                                         line=dict(width=1, color="white")),
                             hovertext=[f"Level: {l}" for l in traj["level"]]))
    fig.update_layout(title="Career trajectory (dot colour = predicted level: red Junior, amber Mid, green Senior)",
                      xaxis_title="Weeks from now", yaxis_title="%", height=420, yaxis=dict(range=[0, 115]))
    return fig


def roi_bar(df):
    fig = px.bar(df, x="roi", y="skill", orientation="h", color="match_gain_pct",
                 color_continuous_scale="Teal", hover_data=["cost_weeks", "weeks_saved", "new_level"])
    fig.update_layout(title="Match gain per learning week (learn the top bar first)",
                      xaxis_title="Match % gained per base week", height=max(260, 42 * len(df) + 110),
                      yaxis=dict(autorange="reversed"))
    return fig


def contrib_bar(df, title, unit):
    colors = ["#ef4444" if c > 0 else "#16a34a" for c in df["contribution"]]
    fig = go.Figure(go.Bar(x=df["contribution"], y=df["factor"], orientation="h", marker_color=colors))
    fig.update_layout(title=title, xaxis_title=unit, yaxis=dict(autorange="reversed"),
                      height=max(260, 40 * len(df) + 110), margin=dict(l=20, r=20, t=50, b=20))
    return fig


def line_chart(df, x, y, title):
    fig = px.line(df, x=x, y=y, markers=True)
    fig.update_layout(title=title, height=340, yaxis=dict(range=[0, 1.05]))
    return fig


def importance_bar(df):
    fig = px.bar(df, x="importance", y="feature", orientation="h")
    fig.update_layout(title="Permutation importance (accuracy drop when a feature is shuffled)",
                      yaxis=dict(autorange="reversed"), height=360)
    return fig