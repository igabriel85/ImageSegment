
exit()
blur = bilateralFilter(foreground, 11, 17, 17)
edged = Canny(blur, 30, 200)
plot_image(edged)
image, contours, hierarchy = findContours(edged.copy(), RETR_TREE,
                                          CHAIN_APPROX_SIMPLE)
print(len(contours))

for cnt in contours:
    print(contourArea(cnt))

    img = foreground  # keypoint[0]
    x, y, w, h = boundingRect(cnt)
    rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)
    plot_image(img)

# drawContours(img, contours, -1, (255, 0, 255), -1)
# plot_image(img)
