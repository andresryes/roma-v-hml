import pandas as pd
import matplotlib.pyplot as plt
import pathlib

file_root = pathlib.Path(__file__).resolve().parent.parent

# Load the CSV
csv_path = file_root / "output" / "eth3d_reconstruction_results.csv"
df = pd.read_csv(csv_path)

df_sorted = df.sort_values(by="Scene")

plt.figure()
for alg, group in df_sorted.groupby("Algorithm"):
    plt.plot(group["Scene"], group["TriangulatedPoints"], marker="o", label=alg)
plt.title("Triangulated Points per Scene")
plt.xlabel("Scene")
plt.ylabel("Triangulated Points")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.legend()
plt.savefig(file_root / "output" / "triangulated_points_per_scene.png")
plt.show()

plt.figure()
for alg, group in df_sorted.groupby("Algorithm"):
    plt.plot(group["Scene"], group["MeanReprojError"], marker="o", label=alg)
plt.title("Mean Reprojection Error per Scene")
plt.xlabel("Scene")
plt.ylabel("Mean Reprojection Error (pixels)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.legend()
plt.savefig(file_root / "output" / "mean_reprojection_error_per_scene.png")
plt.show()

plt.figure()
for alg, group in df.groupby("Algorithm"):
    plt.scatter(group["MatchTime(s)"], group["TriangulatedPoints"], label=alg)
plt.title("Triangulated Points vs. Match Time")
plt.xlabel("Match Time (s)")
plt.ylabel("Triangulated Points")
plt.tight_layout()
plt.legend()
plt.savefig(file_root / "output" / "triangulated_points_vs_match_time.png")
plt.show()