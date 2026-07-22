"""
Visualization: draws bounding boxes color-coded by identification source,
labeled with the matched name and similarity score.
"""

import cv2
import matplotlib.pyplot as plt


def visualize_identities_named(img, identities, show=True):
    """
    Green box  = matched to a known identity.
    Red box    = detected but not matched (Unknown).
    """
    vis_img = img.copy()
    for person in identities:
        x1, y1, x2, y2 = person["bbox"]
        color = (0, 255, 0) if person["name"] != "Unknown" else (0, 0, 255)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        label = f"{person['name']} ({person['similarity']:.2f}, {person['source']})"
        label_y = y1 - 10 if y1 - 10 > 10 else y2 + 20
        cv2.putText(vis_img, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2, cv2.LINE_AA)

    if show:
        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(vis_img_rgb)
        plt.axis("off")
        plt.show()

    return vis_img
