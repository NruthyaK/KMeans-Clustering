# K-Means Clustering: Find the Groups Hiding in the Data

> **Can a machine discover the groups in a dataset without being told the answers?**
> This notebook lets you explore that question by generating synthetic data, clustering it with K-Means, and comparing two popular ways to choose the number of clusters.

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Explore-Jupyter-orange?logo=jupyter&logoColor=white" alt="Jupyter Notebook"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn"></a>
  <img src="https://img.shields.io/badge/Project-Unsupervised%20Learning-7B68EE" alt="Unsupervised learning">
</p>

## The mission

You are given a collection of points, but no cluster labels to train on. Your mission is to:

1. Generate a two-dimensional dataset containing several natural-looking groups.
2. Ask K-Means to group the points using a chosen value of **K**.
3. Visualize the original generated groups and the clusters found by the algorithm.
4. Investigate whether the data supports your choice of K using the **elbow method** and the **silhouette score**.

The notebook's generated data contains four blob centers, and both of its saved K-selection analyses identify **K = 4**. The interesting part is seeing how the algorithm arrives at a useful grouping—and where its assumptions matter.

## Start exploring

Open the notebook and follow it cell by cell:

**[Open `kmeans_1.ipynb`](./kmeans_1.ipynb)**

If GitHub displays the notebook's saved visualizations, you can preview the results right in your browser. To experiment with the code, run it locally in Jupyter or open it in VS Code with the Jupyter extension.

<details>
  <summary><strong>New to clustering? Start here</strong></summary>

  Clustering is an **unsupervised learning** task: the algorithm receives feature values but no answer key telling it which group each observation belongs to. K-Means tries to partition the observations into K groups whose points are close to their assigned cluster center.

  In this notebook, `make_blobs` also returns generated labels (`y`). Those labels are used to color the original groups for a visual comparison; **K-Means itself is fitted only on `X`**, not on `y`.
</details>

## What is inside?

| Notebook stage | What happens | Why it matters |
|---|---|---|
| Create the data | `make_blobs` generates 1,000 samples with two features and four centers, using `random_state=40`. | A simple dataset is easy to inspect in a two-dimensional scatter plot. |
| Choose K | The notebook sets `K = 4` and constructs a `KMeans` estimator. | K-Means needs the number of clusters in advance. |
| Fit and label | `fit_predict(X)` learns cluster centers and assigns each sample a cluster label. | This is the clustering result that can be compared visually. |
| Compare plots | Seaborn scatter plots color the data first by generated labels and then by predicted labels. | A visual comparison makes the discovered structure intuitive. |
| Calculate inertia | K-Means is fitted for K values from 1 through 20, and each model's inertia is saved as WCSS. | Inertia measures the sum of squared distances to each point's assigned center. |
| Find the elbow | `KneeLocator` searches the inertia curve for its bend. | The bend is a practical candidate where adding clusters starts yielding smaller improvements. |
| Calculate silhouette | Silhouette scores are computed for K values from 2 through 20 and plotted. | This evaluates how compact and separated the resulting clusters are. |

## Get set up

You will need Python and Jupyter. From a terminal opened in this project folder, install the notebook's packages:

```bash
python -m pip install jupyter numpy matplotlib seaborn scikit-learn kneed
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open `kmeans_1.ipynb` in the browser, select a Python kernel, and run the cells. You can also open the notebook directly in VS Code if you have the Python and Jupyter extensions installed.

> **Tip:** The notebook includes a cell that installs `kneed`. Installing the listed dependencies before starting is a convenient alternative, especially when working outside Jupyter.

## Your interactive tour

Use these checkpoints as you work through the notebook:

- [ ] **Meet the data:** Run the `make_blobs` cell. What does the shape `(1000, 2)` tell you?
- [ ] **Make your first guess:** Change `K = 4` to another value and rerun the model and scatter plot. What changes?
- [ ] **Inspect the elbow:** Follow the WCSS values for K from 1 to 20. Where does the rate of improvement visibly start to slow?
- [ ] **Check the silhouette:** Run the silhouette calculation and inspect the highest score. Does it agree with the elbow?
- [ ] **Explain your choice:** Write down which K you would choose and what evidence supports it.

<details>
  <summary><strong>Reveal the notebook's saved result</strong></summary>

  - `KneeLocator` reports an elbow at **K = 4**.
  - The highest silhouette score in the notebook's saved list is approximately **0.693** at **K = 4**.
  - This agrees with the four centers used to generate the synthetic blobs.

  The scores are the notebook's saved outputs, not a promise that every rerun will produce identical values: the K-Means estimators do not specify a `random_state`.
</details>

## Two ways to choose K

### The elbow method

The notebook calculates **within-cluster sum of squares (WCSS)**—the K-Means inertia—for each K from 1 to 20. Inertia normally decreases as K increases: with more clusters, points can be closer to their respective centers.

The goal is not simply to choose the K with the lowest inertia. Instead, look for an **elbow**: a bend in the curve after which adding another cluster gives diminishing improvement. `KneeLocator` provides an automatic estimate from the same curve.

### The silhouette method

For each K from 2 to 20, the notebook calculates the mean silhouette score. A silhouette value reflects how close an observation is to its own cluster compared with neighboring clusters; higher overall scores generally suggest better-separated, more cohesive clusters.

**A useful detail:** the silhouette score's mathematical range is **−1 to 1**, not only 0 to 1. Values near 1 indicate well-separated assignments, values near 0 can indicate overlapping clusters, and negative values can indicate that points may be assigned to the wrong cluster.

<details>
  <summary><strong>Quick self-check: what does increasing K do to inertia?</strong></summary>

  Inertia cannot increase just because you allow more clusters: extra centers give the model more flexibility to reduce within-cluster distances. This is why the elbow method looks for a useful trade-off rather than the absolute minimum inertia.
</details>

## Run-order note

The notebook's silhouette plot cell appears **before** the cell that creates `ss`. If you restart the kernel and execute strictly from top to bottom, that plot may fail because `ss` does not exist yet. Run the **Implementation of Silhouette Score** section first, then run the plot cell. The plot is intended to show the scores for K values 2 through 20.

## What to notice—and what not to overclaim

- **K-Means needs K up front.** The notebook demonstrates manual selection and then evaluates possible K values; K-Means does not discover K automatically.
- **The algorithm uses features, not the generated answer key.** The labels `y` come from the synthetic-data generator and are used for plotting only.
- **Cluster IDs are arbitrary.** A cluster named `0` in one run does not necessarily correspond to cluster `0` in another run or to the same-colored original group.
- **Initialization can affect results.** The notebook does not set `random_state` on its `KMeans` estimators, so repeated executions may yield slightly different assignments or scores. The recorded result is one observed run.
- **The example is intentionally friendly.** Blob-shaped groups are a good way to learn K-Means, but real data can be noisy, differently scaled, non-spherical, or contain groups of different sizes. Those conditions can make K-Means less suitable.

## Stretch challenges

Ready to go beyond the guided run? Try one of these:

1. **Make it reproducible:** Set `random_state` on each `KMeans` model, then compare repeated runs.
2. **Change the problem:** Modify the number of generated centers or sample size. Do the elbow and silhouette methods still agree?
3. **Explore a different shape:** Try a dataset that does not form round, well-separated blobs. Observe where K-Means struggles.
4. **Compare evaluation methods:** Run the analysis over a wider or narrower range of K values and explain how your search range affects the conclusion.
5. **Add a visual explanation:** Annotate your plots with the selected K and describe why you chose it.

<details>
  <summary><strong>Want a hint for challenge 1?</strong></summary>

  Pass the same `random_state` value when constructing each `KMeans` estimator. For consistency, apply that change both in the initial clustering cell and inside the loops used to calculate WCSS and silhouette scores.
</details>

## Concepts you'll practice

<details>
  <summary><strong>Show the mini-glossary</strong></summary>

  - **Unsupervised learning:** Finding structure in data without using labeled answers as training targets.
  - **Cluster:** A group of observations considered similar under a chosen distance measure.
  - **Centroid:** The center of a cluster; in K-Means, it is represented by the mean of the assigned points.
  - **Inertia / WCSS:** The sum of squared distances from points to their assigned cluster centers.
  - **K:** The number of clusters requested from K-Means.
  - **Silhouette score:** A measure of how well points fit within their assigned clusters compared with nearby alternatives.
</details>

## Tools used

- **Python** for the notebook's code
- **Jupyter** for interactive, cell-by-cell exploration
- **NumPy** for numerical data
- **scikit-learn** for `make_blobs`, `KMeans`, and `silhouette_score`
- **Seaborn** and **Matplotlib** for visualizations
- **Kneed** for automatic elbow detection with `KneeLocator`

## Repository contents

```text
.
├── README.md
└── kmeans_1.ipynb
```

## Keep exploring

Run the notebook, experiment with K, and see whether the evidence changes your mind. The best K is not just a number—it is a choice you should be able to explain from the data and the purpose of your analysis.

**Ready? [Open the notebook](./kmeans_1.ipynb) and start clustering.**
