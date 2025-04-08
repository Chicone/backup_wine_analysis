from plots import plot_histogram_correlation

# Last 10 channels
# plot_histogram_correlation(
#     "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_remove_10ch_merlot.csv",
#     "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_remove_10ch_cab_sauv.csv",
#     wine1="hist_5ch_greedy_remove_10ch_merlot",
#     wine2="hist_5ch_greedy_remove_10ch_cab_sauv"
# )

# Merlot vs Joint
plot_histogram_correlation(
    "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_remove_M22_merlot_2022.csv",
    "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_remove_M22CS22_merlot_2022.csv",
    wine1="hist_5ch_greedy_remove_M22_merlot_2022",
    wine2="hist_5ch_greedy_remove_M22CS22_merlot_2022"
)

# # CS vs Joint
# plot_histogram_correlation(
#     "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_remove_CS22_cab_sauv_2022.csv",
#     "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_remove_M22CS22_cab_sauv_2022.csv",
#     wine1="hist_5ch_greedy_remove_M22_merlot_2022",
#     wine2="hist_5ch_greedy_remove_M22CS22_merlot_2022"
# )




# plot_histogram_correlation(
#     "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_remove_joint2022_on_merlot2022.csv",
#     "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_remove_joint2022_on_cab_sauv2022.csv",
#     wine1="hist_5ch_greedy_remove_joint2022_on_merlot2022",
#     wine2="hist_5ch_greedy_remove_joint2022_on_cab_sauv2022"
# )