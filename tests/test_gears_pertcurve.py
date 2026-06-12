import pandas as pd

from gears_pertcurve import attach_scores_to_obs, load_cell_scores, set_leakage_free_eval_scores


class FakeAdata:
    def __init__(self):
        self.obs_names = pd.Index(["cell1", "cell2", "cell3"])
        self.obs = pd.DataFrame(index=self.obs_names)


class FakeGraph:
    pass


def test_score_loading_and_attachment(tmp_path):
    csv_path = tmp_path / "projection_A.csv"
    pd.DataFrame(
        {
            "cell_id": ["cell1", "cell2"],
            "normalized_pseudotime": [0.1, 0.8],
            "perturbation": ["control", "A"],
        }
    ).to_csv(csv_path, index=False)

    scores = load_cell_scores([csv_path])
    adata = attach_scores_to_obs(FakeAdata(), scores, default_score=0.0)

    assert list(adata.obs["pertcurve_score"]) == [0.1, 0.8, 0.0]


def test_constant_eval_scores_are_set_on_graphs():
    graphs = [FakeGraph(), FakeGraph()]
    set_leakage_free_eval_scores(graphs, constant_score=0.5)

    assert [g.pertcurve_score for g in graphs] == [0.5, 0.5]
