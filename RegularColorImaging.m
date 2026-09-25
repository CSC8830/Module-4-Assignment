% RegularColorImaging.m
% Classical human boundary detection from an RGB image.
% Assumes the person is the main foreground object and the background is
% visible near the image borders.

inputFile = "person.jpg";
img = imread(inputFile);

if size(img, 3) == 1
    img = repmat(img, [1 1 3]);
end

img = im2uint8(img);
img = imgaussfilt(img, 1);

[rows, cols, ~] = size(img);
grayImg = rgb2gray(img);
labImg = rgb2lab(img);

% Estimate background color from the image border.
borderFrac = 0.06;
border = max(8, round(borderFrac * min(rows, cols)));
bgMask = false(rows, cols);
bgMask(1:border, :) = true;
bgMask(end-border+1:end, :) = true;
bgMask(:, 1:border) = true;
bgMask(:, end-border+1:end) = true;

bgPixels = reshape(labImg, [], 3);
bgPixels = bgPixels(bgMask(:), :);
bgColor = median(bgPixels, 1);

% Foreground likelihood from color distance to background.
colorDist = sqrt(sum((labImg - reshape(bgColor, 1, 1, 3)).^2, 3));
colorDist = mat2gray(colorDist);

% Edge map.
edges = edge(grayImg, 'Canny');
edges = imdilate(edges, strel('disk', 1));

% Threshold the color-distance map.
level = graythresh(colorDist);
maskColor = colorDist > max(0.20, 0.85 * level);

% Clean up the initial mask.
maskColor = imclose(maskColor, strel('disk', 6));
maskColor = imfill(maskColor, 'holes');
maskColor = bwareaopen(maskColor, round(0.002 * numel(maskColor)));

% Merge color and edge cues.
mask = maskColor | edges;
mask = imclose(mask, strel('disk', 6));
mask = imfill(mask, 'holes');
mask = bwareaopen(mask, round(0.003 * numel(mask)));

% Keep the best connected component.
mask = keepBestComponent(mask);

% Refine boundary with active contour.
if any(mask(:))
    mask = activecontour(grayImg, mask, 120, 'edge');
    mask = imclose(mask, strel('disk', 3));
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, round(0.003 * numel(mask)));
    mask = keepBestComponent(mask);
end

boundaries = bwboundaries(mask, 'noholes');

figure;
imshow(img);
hold on;

if ~isempty(boundaries)
    [~, idx] = max(cellfun(@(b) size(b, 1), boundaries));
    b = boundaries{idx};
    plot(b(:, 2), b(:, 1), 'g', 'LineWidth', 2);
end

title('Detected Human Boundary');
hold off;

imwrite(mask, 'humanMask.png');

function mask = keepBestComponent(mask)
cc = bwconncomp(mask);
if cc.NumObjects == 0
    mask = false(size(mask));
    return;
end

stats = regionprops(cc, 'Area', 'Centroid', 'PixelIdxList');
areas = [stats.Area];
centroids = cat(1, stats.Centroid);

[rows, cols] = size(mask);
imageCenter = [cols / 2, rows / 2];

centerDist = hypot(centroids(:,1) - imageCenter(1), centroids(:,2) - imageCenter(2));
score = areas ./ (1 + centerDist);

[~, idx] = max(score);
mask = false(size(mask));
mask(stats(idx).PixelIdxList) = true;
end
