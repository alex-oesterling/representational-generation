import matplotlib.pyplot as plt

# Data for race
# final_relaxed_mprs_race = [
#     0.38737117862646653,
#     0.3740135980448338,
#     0.36065597304508196,
#     0.3472983559565204,
#     0.33394071055788643,
#     0.3205848136985073,
# ]
# final_mprs_race = [
#     0.38737117862646653,
#     0.3780883972744432,
#     0.3666452549567732,
#     0.3422923830517298,
#     0.3345823507426954,
#     0.3259329482542366,
# ]
# final_sims_race = [
#     4.368937075138092,
#     4.366003572940826,
#     4.360307574272156,
#     4.343048110604286,
#     4.3346249759197235,
#     4.320692539215088,
# ]

# # Data for gender (sample data)
# final_relaxed_mprs_gender = [
#     0.40475940749739836,
#     0.38895429649450447,
#     0.37533419328059214,
# ]
# final_mprs_gender = [0.40475940749739836, 0.39617800743436105, 0.3852894034209037]
# final_sims_gender = [4.668411761522293, 4.663244411349297, 4.649248704314232]


final_relaxed_mprs_race = [0.0959520920260877]
final_mprs_race = [0.0959520920260877]
final_sims_race = [4.368937075138092]
# Data for gender (sample data)
final_relaxed_mprs_gender = [
    0.09185373437124195,
    0.0880296833923576,
    0.08401455730655157,
]
final_mprs_gender = [0.09185373437124195, 0.08871645778149194, 0.08439862984763244]
final_sims_gender = [4.668411761522293, 4.658641412854195, 4.637324571609497]
# Generate x values (e.g., epochs or run indices)
x_race = list(range(1, len(final_relaxed_mprs_race) + 1))
x_gender = list(range(1, len(final_relaxed_mprs_gender) + 1))

# Plot Final Relaxed MPRs
plt.figure(figsize=(12, 6))
plt.plot(
    x_race,
    final_relaxed_mprs_race,
    marker="o",
    label="Race Final Relaxed MPRs",
    color="red",
)
plt.plot(
    x_gender,
    final_relaxed_mprs_gender,
    marker="x",
    label="Gender Final Relaxed MPRs",
    color="blue",
)
plt.xlabel("Run Index")
plt.ylabel("Relaxed MPRs")
plt.title("Final Relaxed MPRs Over Different Runs")
plt.legend()
plt.grid(True)
plt.show()

# Plot Final MPRs
plt.figure(figsize=(12, 6))
plt.plot(x_race, final_mprs_race, marker="o", label="Race Final MPRs", color="red")
plt.plot(
    x_gender, final_mprs_gender, marker="x", label="Gender Final MPRs", color="blue"
)
plt.xlabel("Run Index")
plt.ylabel("MPRs")
plt.title("Final MPRs Over Different Runs")
plt.legend()
plt.grid(True)
plt.show()

# Plot Final Similarities
plt.figure(figsize=(12, 6))
plt.plot(
    x_race, final_sims_race, marker="o", label="Race Final Similarities", color="red"
)
plt.plot(
    x_gender,
    final_sims_gender,
    marker="x",
    label="Gender Final Similarities",
    color="blue",
)
plt.xlabel("Run Index")
plt.ylabel("Similarities")
plt.title("Final Similarities Over Different Runs")
plt.legend()
plt.grid(True)
plt.show()
