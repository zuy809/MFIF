function [R1_all, R2_all, wmap_all] = mfif_fuse(input_path, output_path, group_size, cfg)
%MFIF_FUSE
close all;

if nargin < 1 || isempty(input_path)
    error('input_path is required.');
end
if nargin < 2 || isempty(output_path)
    output_path = input_path;
end
if nargin < 3 || isempty(group_size)
    group_size = 10;
end
if nargin < 4 || isempty(cfg)
    cfg = struct();
end

p = merge_params(base_params(), cfg);
input_path = char(input_path);
output_path = char(output_path);

group_path = fullfile(output_path, 'group_sources');
result_path = fullfile(output_path, 'result');
map_path = fullfile(output_path, 'dmap');
if ~exist(group_path, 'dir'), mkdir(group_path); end
if ~exist(result_path, 'dir'), mkdir(result_path); end
if ~exist(map_path, 'dir'), mkdir(map_path); end

fprintf('===== Step 1: load images =====\n');
fprintf('    input_path: %s\n', input_path);
fprintf('    output_path: %s\n', output_path);
fprintf('    group_size: %d\n', group_size);

files = dir(fullfile(input_path, '*.tif'));
if isempty(files)
    files = dir(fullfile(input_path, '*.tiff'));
end
if isempty(files)
    error('No tif images found in input_path.');
end

ids = zeros(numel(files), 1);
for i = 1:numel(files)
    [~, name, ~] = fileparts(files(i).name);
    nums = regexp(name, '\d+', 'match');
    if isempty(nums)
        ids(i) = i;
    else
        ids(i) = str2double(nums{1});
    end
end
[~, order] = sort(ids);
files = files(order);

n = numel(files);
imgs = cell(n, 1);
gray = cell(n, 1);
for i = 1:n
    img = imread(fullfile(input_path, files(i).name));
    img = to_rgb8(img);
    imgs{i} = img;
    gray{i} = double(rgb2gray(img));
end

fprintf('===== Step 2: group fusion =====\n');
tic;
group_size = max(1, round(double(group_size)));
num_groups = ceil(n / group_size);
groups = cell(num_groups, 1);
R1_all = cell(max(num_groups - 1, 0), 1);
R2_all = cell(max(num_groups - 1, 0), 1);
wmap_all = cell(max(num_groups - 1, 0), 1);
for g = 1:num_groups
    s = (g - 1) * group_size + 1;
    e = min(g * group_size, n);
    groups{g} = group_fuse(gray(s:e), imgs(s:e), p);
    imwrite(groups{g}, fullfile(group_path, sprintf('group_%02d.tif', g)));
    fprintf('    saved group_%02d.tif (%d-%d)\n', g, s, e);
end

fprintf('===== Step 3: pair fusion =====\n');
cur = groups{1};
for g = 2:num_groups
    tag = sprintf('Inter_G%02d', g);
    [cur, R1, R2, wmap] = pair_fuse(cur, groups{g}, map_path, tag, p);
    slot = g - 1;
    R1_all{slot} = R1;
    R2_all{slot} = R2;
    wmap_all{slot} = wmap;
    save_maps(R1, R2, map_path, tag);
    imwrite(cur, fullfile(result_path, sprintf('intermediate_fused_%02d.tif', g)));
end

cur = source_consistency(cur, gray);

imwrite(cur, fullfile(result_path, 'final_fused_result.tif'));
fprintf('===== Done: %.4f seconds =====\n', toc);
end

function p = base_params()
p = struct();
p.it = 2;
p.win = [13 3];
p.ct = 0.9;
p.gcr = 5;
p.gr = 5;
p.ge = 0.06;
p.gs = 0.1;
p.gk = 5;
p.t = 0.02;
p.tt = 0.4;
p.ar = 0.01;
p.bt = 5;
p.rOff = 0.1;
end

function p = merge_params(p, cfg)
if ~isstruct(cfg)
    error('cfg must be a struct.');
end

aliases = {
    'numIterations', 'it';
    'winSizes', 'win';
    'corrThreshold', 'ct';
    'groupCorrRadius', 'gcr';
    'guidedRadius', 'gr';
    'guidedEps', 'ge';
    'gaussSigma', 'gs';
    'gaussFilterSize', 'gk';
    'trimapT', 't';
    'TT', 'tt';
    'areaRatio', 'ar';
    'blackProtectThreshold', 'bt'
};

for i = 1:size(aliases, 1)
    src = aliases{i, 1};
    dst = aliases{i, 2};
    if isfield(cfg, src) && ~isempty(cfg.(src))
        p.(dst) = cfg.(src);
    end
end

names = fieldnames(cfg);
for i = 1:numel(names)
    name = names{i};
    if isfield(p, name) && ~isempty(cfg.(name))
        p.(name) = cfg.(name);
    end
end

p.it = max(1, round(double(p.it)));
if numel(p.win) < p.it
    p.win = repmat(p.win(1), 1, p.it);
end
p.win = p.win(1:p.it);
p.gcr = max(1, round(double(p.gcr)));
p.gr = max(1, round(double(p.gr)));
p.gk = max(3, round(double(p.gk)));
if mod(p.gk, 2) == 0
    p.gk = p.gk + 1;
end
end

function img = to_rgb8(img)
if ~isa(img, 'uint8')
    img = im2uint8(img);
end
if ismatrix(img)
    img = repmat(img, [1 1 3]);
elseif size(img, 3) > 3
    img = img(:, :, 1:3);
end
end

function save_maps(R1, R2, save_path, tag)
R1_vis = R1;
R1_vis(R1_vis < 0) = 0;
R1_vis = min(R1_vis / 0.1, 1);

R2_vis = R2;
R2_vis(R2_vis < 0) = 0;
R2_vis = min(R2_vis / 0.1, 1);

imwrite(R1_vis, fullfile(save_path, [tag '_R1_map.png']));
imwrite(R2_vis, fullfile(save_path, [tag '_R2_map.png']));
end

function [fused, R1, R2, wmap] = pair_fuse(I1_orig, I2_orig, save_path, tag, p)
I1_orig = to_rgb8(I1_orig);
I2_orig = to_rgb8(I2_orig);
I1 = double(rgb2gray(I1_orig));
I2 = double(rgb2gray(I2_orig));
D = (I1 + I2) / 2;

for k = 1:p.it
    win = p.win(k);
    R1_temp = fast_corr(D, I1, win, 1e-10);
    R2_temp = fast_corr(D, I2, win, 1e-10);

    S1 = imgaussfilt(R1_temp, p.gs, 'FilterSize', p.gk);
    S2 = imgaussfilt(R2_temp, p.gs, 'FilterSize', p.gk);
    w = double((S1 .* (S1 > p.ct)) > (S2 .* (S2 > p.ct)));

    w(I2 < p.bt & I1 >= p.bt) = 1;
    w(I1 < p.bt & I2 >= p.bt) = 0;

    G = I1 / 255 .* w + I2 / 255 .* (1 - w);
    w = guidedfilter_lkh(G, w, p.gr, p.ge);

    tri = make_trimap(w, p.t);
    fixed_map = tri ~= 0.5;
    fixed_vals = double(tri == 1);
    alpha = close_matte(I1_orig, fixed_map, fixed_vals);
    N = boxfilter(ones(size(I1)), p.gr);
    alpha = boxfilter(alpha, p.gr) ./ N;
    w = alpha >= p.tt;

    area = ceil(p.ar * numel(w));
    m1 = bwareaopen(w, area);
    m2 = 1 - m1;
    m3 = bwareaopen(m2, area);
    w = 1 - m3;

    D = I1 .* w + I2 .* (1 - w);

    if k == p.it
        R1 = R1_temp - p.rOff;
        R2 = R2_temp - p.rOff;
        wmap = w;
    end
end

w_up = imresize(w, [size(I1_orig, 1), size(I1_orig, 2)]);
imwrite(w_up, fullfile(save_path, [tag '_weight.png']));
fused = uint8(double(I1_orig) .* repmat(w_up, [1 1 3]) + ...
    double(I2_orig) .* repmat(1 - w_up, [1 1 3]));
end

function tri = make_trimap(w, t)
tri = zeros(size(w));
tri(w > 1 - t) = 1;
tri(w < t) = 0;
tri(w >= t & w <= 1 - t) = 0.5;
end

function corr_map = fast_corr(img1, img2, r, eps_ecc)
[h, w] = size(img1);
N = boxfilter(ones(h, w), r);
m1 = boxfilter(img1, r) ./ N;
m2 = boxfilter(img2, r) ./ N;
cov12 = boxfilter(img1 .* img2, r) ./ N - m1 .* m2;
v1 = max(0, boxfilter(img1 .* img1, r) ./ N - m1 .* m1);
v2 = max(0, boxfilter(img2 .* img2, r) ./ N - m2 .* m2);
corr_map = cov12 ./ (sqrt(v1) .* sqrt(v2) + eps_ecc);
corr_map = max(-1, min(1, corr_map));
end

function imDst = boxfilter(imSrc, r)
[hei, wid] = size(imSrc);
imDst = zeros(size(imSrc));
imCum = cumsum(imSrc, 1);
imDst(1:r + 1, :) = imCum(1 + r:2 * r + 1, :);
imDst(r + 2:hei - r, :) = imCum(2 * r + 2:hei, :) - imCum(1:hei - 2 * r - 1, :);
imDst(hei - r + 1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei - 2 * r:hei - r - 1, :);
imCum = cumsum(imDst, 2);
imDst(:, 1:r + 1) = imCum(:, 1 + r:2 * r + 1);
imDst(:, r + 2:wid - r) = imCum(:, 2 * r + 2:wid) - imCum(:, 1:wid - 2 * r - 1);
imDst(:, wid - r + 1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid - 2 * r:wid - r - 1);
end

function q = guidedfilter_lkh(I, p, r, eps_val)
[hei, wid] = size(I);
N = boxfilter(ones(hei, wid), r);
mI = boxfilter(I, r) ./ N;
mp = boxfilter(p, r) ./ N;
mIp = boxfilter(I .* p, r) ./ N;
covIp = mIp - mI .* mp;
mII = boxfilter(I .* I, r) ./ N;
varI = mII - mI .* mI;
a = covIp ./ (varI + eps_val);
b = mp - a .* mI;
q = (boxfilter(a, r) ./ N) .* I + (boxfilter(b, r) ./ N);
end

function fused = group_fuse(gray_set, img_set, p)
num_imgs = numel(gray_set);
avg = zeros(size(gray_set{1}));
for i = 1:num_imgs
    avg = avg + gray_set{i};
end
avg = avg / num_imgs;

[h, w] = size(avg);
corr_maps = zeros(h, w, num_imgs);
for i = 1:num_imgs
    corr_maps(:, :, i) = fast_corr(avg, gray_set{i}, p.gcr, 1e-10);
end
[~, pick] = max(corr_maps, [], 3);

[oh, ow, ~] = size(img_set{1});
pick = imresize(pick, [oh, ow], 'nearest');
fused = zeros(oh, ow, 3, 'uint8');
for i = 1:num_imgs
    mask = repmat(pick == i, [1 1 3]);
    src = to_rgb8(img_set{i});
    fused(mask) = src(mask);
end
end

function refined = source_consistency(fused, source_gray)
mix = 0;
block_size = 16;
feather = true;
if mix <= 0
    refined = fused;
    return;
end

if ndims(fused) == 3
    fused_gray = double(rgb2gray(fused));
else
    fused_gray = double(fused);
end
[h, w] = size(fused_gray);
num_src = numel(source_gray);
stack = zeros(h, w, num_src);
for k = 1:num_src
    src = double(source_gray{k});
    if ~isequal(size(src), [h, w])
        src = imresize(src, [h, w]);
    end
    stack(:, :, k) = src;
end

flt = [-1 0 1; -2 0 2; -1 0 1];
target = fused_gray;
num_h = ceil(h / block_size);
num_w = ceil(w / block_size);
score = zeros(num_src, 1);
for bh = 1:num_h
    y1 = (bh - 1) * block_size + 1;
    y2 = min(bh * block_size, h);
    for bw = 1:num_w
        x1 = (bw - 1) * block_size + 1;
        x2 = min(bw * block_size, w);
        for k = 1:num_src
            patch = stack(y1:y2, x1:x2, k);
            gx = filter2(flt, patch, 'same');
            gy = filter2(flt', patch, 'same');
            grad = sqrt(gx .^ 2 + gy .^ 2);
            score(k) = sum(grad(:).^5);
        end
        [~, idx] = max(score);
        target(y1:y2, x1:x2) = stack(y1:y2, x1:x2, idx);
    end
end

if feather
    target = imgaussfilt(target, 0.35, 'FilterSize', 3);
end

refined_gray = (1 - mix) * fused_gray + mix * target;
refined_gray = uint8(max(0, min(255, round(refined_gray))));
refined = repmat(refined_gray, [1 1 3]);
end

function alpha = close_matte(I, fixed_map, fixed_vals)
I = double(to_rgb8(I)) / 255;
alpha = solve_alpha(I, fixed_map, fixed_vals);
end

function alpha = solve_alpha(I, fixed_map, fixed_vals)
[h, w, ~] = size(I);
img_size = w * h;
A = get_laplacian(I, fixed_map);
D = spdiags(fixed_map(:), 0, img_size, img_size);
lambda = 100;
x = (A + lambda * D) \ (lambda * fixed_map(:) .* fixed_vals(:));
alpha = max(min(reshape(x, h, w), 1), 0);
end

function A = get_laplacian(I, fixed_map)
epsilon = 1e-7;
win_size = 1;
neb_size = (win_size * 2 + 1) ^ 2;
[h, w, c] = size(I);
img_size = w * h;
fixed_map = imerode(fixed_map, ones(win_size * 2 + 1));
inds = reshape(1:img_size, h, w);
max_entries = max(0, (h - 2 * win_size) * (w - 2 * win_size) * neb_size ^ 2);
row = zeros(max_entries, 1);
col = zeros(max_entries, 1);
val = zeros(max_entries, 1);
pos = 0;
for j = 1 + win_size:w - win_size
    for i = 1 + win_size:h - win_size
        if fixed_map(i, j), continue; end
        win_inds = inds(i - win_size:i + win_size, j - win_size:j + win_size);
        win_inds = win_inds(:);
        winI = I(i - win_size:i + win_size, j - win_size:j + win_size, :);
        winI = reshape(winI, neb_size, c);
        win_mu = mean(winI, 1)';
        cov_mat = winI' * winI / neb_size - win_mu * win_mu' + ...
            epsilon / neb_size * eye(c);
        win_var = cov_mat \ eye(c);
        winI = winI - repmat(win_mu', neb_size, 1);
        tvals = (1 + winI * win_var * winI') / neb_size;
        idx = pos + (1:neb_size ^ 2);
        row(idx) = reshape(repmat(win_inds, 1, neb_size), neb_size ^ 2, 1);
        col(idx) = reshape(repmat(win_inds', neb_size, 1), neb_size ^ 2, 1);
        val(idx) = tvals(:);
        pos = pos + neb_size ^ 2;
    end
end
row = row(1:pos);
col = col(1:pos);
val = val(1:pos);
A = sparse(row, col, val, img_size, img_size);
A = spdiags(sum(A, 2), 0, img_size, img_size) - A;
end
