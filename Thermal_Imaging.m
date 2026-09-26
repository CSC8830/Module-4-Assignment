% Thermal_Imaging.m
% Segment a human in a thermal image and extract the boundary.

inputFile = "person.jpg";

I0 = imread(inputFile);
if ndims(I0) == 3
    I0 = rgb2gray(I0);
end

I = im2single(I0);
I = mat2gray(I);
I = medfilt2(I, [3 3]);
I = adapthisteq(I, "ClipLimit", 0.01, "Distribution", "rayleigh");
I = imgaussfilt(I, 1);

sensitivities = [0.30 0.40 0.50 0.60 0.70];
polarities = ["bright" "dark"];

bestMask = false(size(I));
bestScore = -Inf;

for s = sensitivities
    for p = polarities
        bw = imbinarize(I, "adaptive", ...
            "ForegroundPolarity", p, ...
            "Sensitivity", s);

        bw = cleanMask(bw);
        bw = selectBestComponent(bw);

        score = scoreMask(bw);
        if score > bestScore
            bestScore = score;
            bestMask = bw;
        end
    end
end

mask = bestMask;

boundaries = bwboundaries(mask, "noholes");
boundary = [];

if ~isempty(boundaries)
    [~, idx] = max(cellfun(@(c) size(c, 1), boundaries));
    boundary = boundaries{idx};
end

figure;
imshow(I, []);
hold on;

if ~isempty(boundary)
    plot(boundary(:, 2), boundary(:, 1), "r", "LineWidth", 1.5);
end

title("Thermal human boundary");
imwrite(mask, "humanMask.png");

function bw = cleanMask(bw)
bw = logical(bw);
bw = bwareaopen(bw, max(50, round(numel(bw) * 0.002)));
bw = imclose(bw, strel("disk", 3));
bw = imopen(bw, strel("disk", 1));
bw = imfill(bw, "holes");
end

function bw = selectBestComponent(bw)
bw = logical(bw);
cc = bwconncomp(bw);

if cc.NumObjects == 0
    bw = false(size(bw));
    return;
end

stats = regionprops(cc, "Area", "Centroid", "Perimeter", "BoundingBox");
[rows, cols] = size(bw);
imageCenter = [(cols + 1) / 2, (rows + 1) / 2];

bestScore = -Inf;
bestIdx = 1;

for k = 1:cc.NumObjects
    area = stats(k).Area;
    centroid = stats(k).Centroid;
    perimeter = stats(k).Perimeter;
    bbox = stats(k).BoundingBox;

    touchesBorder = bbox(1) <= 1 || bbox(2) <= 1 || ...
        (bbox(1) + bbox(3)) >= cols || (bbox(2) + bbox(4)) >= rows;

    dist = norm(centroid - imageCenter) / norm([cols, rows] / 2);
    compactness = 4 * pi * area / (perimeter^2 + eps);
    areaFrac = area / numel(bw);
    aspect = bbox(4) / (bbox(3) + eps);

    score = log(area + 1) + 2.5 * compactness - 2 * dist - 4 * touchesBorder;
    score = score + 0.2 * min(aspect, 5);
    score = score - 3 * max(0, areaFrac - 0.45) - 1 * max(0, 0.01 - areaFrac);

    if score > bestScore
        bestScore = score;
        bestIdx = k;
    end
end

bw = false(size(bw));
bw(cc.PixelIdxList{bestIdx}) = true;
end

function s = scoreMask(bw)
cc = bwconncomp(bw);

if cc.NumObjects == 0
    s = -Inf;
    return;
end

stats = regionprops(cc, "Area", "Centroid", "Perimeter", "BoundingBox");
area = stats(1).Area;
centroid = stats(1).Centroid;
perimeter = stats(1).Perimeter;
bbox = stats(1).BoundingBox;

[rows, cols] = size(bw);
imageCenter = [(cols + 1) / 2, (rows + 1) / 2];
dist = norm(centroid - imageCenter) / norm([cols, rows] / 2);

touchesBorder = bbox(1) <= 1 || bbox(2) <= 1 || ...
    (bbox(1) + bbox(3)) >= cols || (bbox(2) + bbox(4)) >= rows;

compactness = 4 * pi * area / (perimeter^2 + eps);

s = log(area + 1) + 2.5 * compactness - 2 * dist - 4 * touchesBorder;
end
