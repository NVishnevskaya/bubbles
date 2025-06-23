import lib_bubble

if __name__ == "__main__":
    r = 3
    dots = lib_bubble.build_net(35, 35, r, 0.1 * r, 10)
    lib_bubble.show_net(dots, r, "На связи",)
    kmeans_preds = lib_bubble.get_kmeans_prediction(dots, 10)
    lib_bubble.show_net_with_polygons(dots, R=r, bubble_colors=kmeans_preds)
    dbscan_preds = lib_bubble.get_dbscan_prediction(dots, r * 2.2, 5,
                                                    True)
    lib_bubble.show_net_with_polygons(dots, R=r, bubble_colors=dbscan_preds,
                                      optimize=True, to_expand_dots=True,
                                      save_to="dbscan.png", title="DBSCAN*")