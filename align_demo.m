function results = align_demo_residual(image_dir, output_dir, varargin)
%ALIGN_DEMO_RESIDUAL Residual cascade registration for multi-focus sequences.
%
% Usage in MATLAB:
%   cd('path/to/this/repository');
%   results = align_demo_residual();
%   results = align_demo_residual(input_dir, output_dir, 'noi', 100, 'levels', 3);
%
% Difference from align_demo.m:
%   1) estimate only adjacent residual transforms on original images:
%        I49 -> I50, I48 -> I49, ...
%   2) compose those transforms to the last reference frame;
%   3) warp every original image only once to the last-frame coordinate.

    clc; close all;
    repo_dir = fileparts(mfilename('fullpath'));

    if nargin < 1 || isempty(image_dir)
        image_dir = fullfile(repo_dir, 'dataset', 'ball');
    end
    if nargin < 2 || isempty(output_dir)
        output_dir = fullfile(repo_dir, 'outputs', 'ball_aligned');
    end

    config.image_dir  = image_dir;
    config.output_dir = output_dir;

    config.levels    = 3;
    config.noi       = 100;
    config.transform = 'affine';

    % Adjacent frames in multi-focus data often have rho around 0.90+.
    % 0.99 is too strict here and would mark many usable residual transforms as failed.
    config.min_rho = 0.85;
    config.max_translation = 1000.0;
    config.min_det = 0.0;
    config.max_det = 10.0;

    % Use the previous adjacent warp as the initial value for the next adjacent pair.
    % This matches the residual idea: previous motion is a prior, not a warped image.
    config.use_previous_pair_init = true;
    config = apply_name_value_options(config, varargin{:});

    if ~exist(config.output_dir, 'dir')
        mkdir(config.output_dir);
    end

    image_paths = get_image_files(config.image_dir);
    n_images = numel(image_paths);
    if n_images < 2
        error('Need at least 2 images in config.image_dir');
    end

    fprintf('Residual registration\n');
    fprintf('  Input : %s\n', config.image_dir);
    fprintf('  Output: %s\n', config.output_dir);
    fprintf('  Images: %d\n', n_images);
    fprintf('  Transform: %s, levels=%d, noi=%d\n\n', ...
        config.transform, config.levels, config.noi);

    ref_idx = n_images;
    ref_img = force_rgb_uint8(imread(image_paths{ref_idx}));
    [ref_h, ref_w, ref_c] = size(ref_img);

    total_warps = cell(n_images, 1);
    adjacent_warps = cell(n_images, 1);
    total_warps{ref_idx} = eye(3);

    pair_results = repmat(empty_pair_result(), n_images - 1, 1);
    previous_pair_warp = [];

    % Step 1: estimate adjacent residual transforms on original neighboring frames.
    fprintf('===== Step 1: estimate adjacent residual transforms =====\n');
    for i = (ref_idx - 1):-1:1
        moving = force_rgb_uint8(imread(image_paths{i}));
        template = force_rgb_uint8(imread(image_paths{i + 1}));

        init_warp = [];
        if config.use_previous_pair_init && ~isempty(previous_pair_warp)
            init_warp = previous_pair_warp;
        end

        t0 = tic;
        [adjacent_warp, stats] = estimate_adjacent_warp(moving, template, config, init_warp);
        elapsed = toc(t0);

        adjacent_warps{i} = adjacent_warp;
        total_warps{i} = compose_warps(adjacent_warp, total_warps{i + 1});

        if stats.success
            previous_pair_warp = adjacent_warp;
        else
            previous_pair_warp = [];
        end

        [~, moving_name, moving_ext] = fileparts(image_paths{i});
        [~, template_name, template_ext] = fileparts(image_paths{i + 1});
        pair_results(i) = struct( ...
            'index', i, ...
            'moving_file', [moving_name moving_ext], ...
            'template_file', [template_name template_ext], ...
            'rho', stats.rho, ...
            'mse_before', stats.mse_before, ...
            'mse_after', stats.mse_after, ...
            'improvement', stats.improvement, ...
            'translation', stats.translation, ...
            'det_value', stats.det_value, ...
            'success', stats.success, ...
            'elapsed_sec', elapsed);

        fprintf('  %03d -> %03d | rho=%.6f, trans=%.3f, det=%.6f, success=%d, %.2fs\n', ...
            i, i + 1, stats.rho, stats.translation, stats.det_value, stats.success, elapsed);

        clear moving template;
    end

    % Step 2: warp each original image once to the final reference coordinate.
    fprintf('\n===== Step 2: warp originals once to reference frame %03d =====\n', ref_idx);
    for i = 1:n_images
        img = force_rgb_uint8(imread(image_paths{i}));

        if i == ref_idx
            result_img = ref_img;
        else
            result_img = apply_warp_to_reference(img, total_warps{i}, config.transform, ref_h, ref_w, ref_c);
        end

        out_name = fullfile(config.output_dir, sprintf('%03d.tif', i));
        imwrite(result_img, out_name);
        fprintf('  saved %03d.tif\n', i);

        clear img result_img;
    end

    % Step 3: save diagnostics.
    pair_table = struct2table(pair_results);
    pair_log_path = fullfile(config.output_dir, 'residual_pairwise_log.csv');
    writetable(pair_table, pair_log_path);

    source_table = build_source_table(image_paths);
    source_map_path = fullfile(config.output_dir, 'source_index_map.csv');
    writetable(source_table, source_map_path);

    save(fullfile(config.output_dir, 'residual_alignment_results.mat'), ...
        'config', 'image_paths', 'ref_idx', 'adjacent_warps', 'total_warps', 'pair_table', 'source_table');

    results = struct( ...
        'config', config, ...
        'image_paths', {image_paths}, ...
        'reference_idx', ref_idx, ...
        'adjacent_warps', {adjacent_warps}, ...
        'total_warps', {total_warps}, ...
        'pair_table', pair_table, ...
        'source_table', source_table);

    fprintf('\nDone.\n');
    fprintf('  Pair log : %s\n', pair_log_path);
    fprintf('  Map log  : %s\n', source_map_path);
end

function row = empty_pair_result()
    row = struct( ...
        'index', 0, ...
        'moving_file', '', ...
        'template_file', '', ...
        'rho', NaN, ...
        'mse_before', NaN, ...
        'mse_after', NaN, ...
        'improvement', NaN, ...
        'translation', NaN, ...
        'det_value', NaN, ...
        'success', false, ...
        'elapsed_sec', NaN);
end

function [warp_matrix, stats] = estimate_adjacent_warp(moving_color, template_color, config, init_warp)
    moving_gray = rgb2gray(moving_color);
    template_gray = rgb2gray(template_color);

    moving_d = double(moving_color);
    template_d = double(template_color);
    mse_before = mean((moving_d(:) - template_d(:)).^2);

    try
        if isempty(init_warp)
            [ecc_results, warp] = ecc( ...
                moving_gray, template_gray, ...
                config.levels, config.noi, config.transform);
        else
            init_arg = matrix_to_ecc_init(init_warp, config.transform);
            [ecc_results, warp] = ecc( ...
                moving_gray, template_gray, ...
                config.levels, config.noi, config.transform, init_arg);
        end

        warp_matrix = warp_to_matrix(warp, config.transform);
        rho = ecc_results(1, end).rho;

        warped_moving = apply_warp_to_reference(moving_color, warp_matrix, ...
            config.transform, size(template_color, 1), size(template_color, 2), size(template_color, 3));
        warped_d = double(warped_moving);
        mse_after = mean((warped_d(:) - template_d(:)).^2);

        if mse_before > 0
            improvement = (mse_before - mse_after) / mse_before * 100;
        else
            improvement = 0;
        end

        [translation, det_value, reasonable] = transform_reasonable(warp_matrix, config);
        success = (rho >= config.min_rho) && reasonable;
    catch ME
        warning('Residual alignment failed: %s', ME.message);
        warp_matrix = eye(3);
        rho = 0;
        mse_after = mse_before;
        improvement = 0;
        translation = 0;
        det_value = 1;
        success = false;
    end

    stats = struct( ...
        'rho', rho, ...
        'mse_before', mse_before, ...
        'mse_after', mse_after, ...
        'improvement', improvement, ...
        'translation', translation, ...
        'det_value', det_value, ...
        'success', success);
end

function total = compose_warps(adjacent, next_total)
% spatial_interp samples input at warp * output_coordinate.
% Therefore:
%   W_i_to_final = W_i_to_i+1 * W_i+1_to_final
    total = adjacent * next_total;
    total(3, :) = [0 0 1];
end

function result_img = apply_warp_to_reference(img, warp_matrix, transform, ref_h, ref_w, ref_c)
    img = force_rgb_uint8(img);

    if size(img, 3) == 1 && ref_c == 3
        img = repmat(img, [1 1 3]);
    end

    result_img = zeros(ref_h, ref_w, size(img, 3), 'uint8');
    spatial_warp = matrix_to_spatial_warp(warp_matrix, transform);

    for ch = 1:size(img, 3)
        warped_ch = spatial_interp( ...
            double(img(:, :, ch)), ...
            spatial_warp, ...
            'linear', ...
            transform, ...
            1:ref_w, ...
            1:ref_h);
        warped_ch = min(max(warped_ch, 0), 255);
        result_img(:, :, ch) = uint8(warped_ch);
    end
end

function matrix = warp_to_matrix(warp, transform)
    if strcmp(transform, 'translation')
        matrix = [1 0 warp(1); 0 1 warp(2); 0 0 1];
    elseif strcmp(transform, 'affine') || strcmp(transform, 'euclidean')
        matrix = double(warp);
        if size(matrix, 1) == 2
            matrix = [matrix; 0 0 1];
        else
            matrix(3, :) = [0 0 1];
        end
    elseif strcmp(transform, 'homography')
        matrix = double(warp);
        matrix = matrix ./ matrix(3, 3);
    else
        error('Unsupported transform: %s', transform);
    end
end

function warp = matrix_to_spatial_warp(matrix, transform)
    if strcmp(transform, 'translation')
        warp = matrix(1:2, 3);
    elseif strcmp(transform, 'affine') || strcmp(transform, 'euclidean')
        warp = matrix;
    elseif strcmp(transform, 'homography')
        warp = matrix ./ matrix(3, 3);
    else
        error('Unsupported transform: %s', transform);
    end
end

function init_arg = matrix_to_ecc_init(matrix, transform)
    if strcmp(transform, 'translation')
        init_arg = matrix(1:2, 3);
    elseif strcmp(transform, 'affine') || strcmp(transform, 'euclidean')
        init_arg = matrix(1:2, :);
    elseif strcmp(transform, 'homography')
        init_arg = matrix ./ matrix(3, 3);
    else
        error('Unsupported transform: %s', transform);
    end
end

function [translation, det_value, reasonable] = transform_reasonable(matrix, config)
    translation = hypot(matrix(1, 3), matrix(2, 3));
    det_value = det(matrix(1:2, 1:2));

    if strcmp(config.transform, 'affine') || strcmp(config.transform, 'euclidean')
        reasonable = (translation < config.max_translation) && ...
            (det_value > config.min_det) && ...
            (det_value < config.max_det);
    elseif strcmp(config.transform, 'translation')
        reasonable = (translation < config.max_translation);
    else
        reasonable = true;
    end
end

function img = force_rgb_uint8(img)
    if ~isa(img, 'uint8')
        img = im2uint8(img);
    end
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
end

function image_paths = get_image_files(image_dir)
    patterns = {'*.bmp','*.jpg','*.jpeg','*.png','*.tiff','*.tif'};
    all_files = {};
    for i = 1:numel(patterns)
        f1 = dir(fullfile(image_dir, patterns{i}));
        for j = 1:numel(f1)
            all_files{end + 1} = fullfile(image_dir, f1(j).name);
        end
        f2 = dir(fullfile(image_dir, upper(patterns{i})));
        for j = 1:numel(f2)
            all_files{end + 1} = fullfile(image_dir, f2(j).name);
        end
    end

    all_files = unique(all_files);
    image_paths = natural_sort(all_files);
end

function sorted_list = natural_sort(file_list)
    [~, names, ~] = cellfun(@fileparts, file_list, 'UniformOutput', false);
    nums = zeros(numel(names), 1);
    for i = 1:numel(names)
        tokens = regexp(names{i}, '\d+', 'match');
        if ~isempty(tokens)
            nums(i) = str2double(tokens{end});
        else
            nums(i) = i;
        end
    end
    [~, idx] = sort(nums);
    sorted_list = file_list(idx);
end

function source_table = build_source_table(image_paths)
    n = numel(image_paths);
    output_index = (1:n)';
    source_file = cell(n, 1);
    source_path = cell(n, 1);

    for i = 1:n
        [~, name, ext] = fileparts(image_paths{i});
        source_file{i} = [name ext];
        source_path{i} = image_paths{i};
    end

    source_table = table(output_index, source_file, source_path);
end

function config = apply_name_value_options(config, varargin)
    if mod(numel(varargin), 2) ~= 0
        error('Optional arguments must be name-value pairs.');
    end

    for k = 1:2:numel(varargin)
        name = lower(string(varargin{k}));
        value = varargin{k + 1};

        switch name
            case "levels"
                config.levels = value;
            case "noi"
                config.noi = value;
            case "transform"
                config.transform = lower(char(value));
            case "minrho"
                config.min_rho = value;
            case "min_rho"
                config.min_rho = value;
            case "maxtranslation"
                config.max_translation = value;
            case "max_translation"
                config.max_translation = value;
            case "usedpreviouspairinit"
                config.use_previous_pair_init = logical(value);
            case "use_previous_pair_init"
                config.use_previous_pair_init = logical(value);
            otherwise
                error('Unknown option: %s', varargin{k});
        end
    end
end
