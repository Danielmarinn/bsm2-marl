function filter_and_rename_csv(input_file, output_file, cols_to_keep, new_names)
% FILTER_AND_RENAME_CSV
%   Read a CSV, select specific columns, rename them and save a new CSV.
%
%   Usage:
%       filter_and_rename_csv( ...
%           'raw_state.csv', ...
%           'state.csv', ...
%           {'S_NO', 'S_NH'}, ...
%           {'SNO_anox', 'SNH_in'})
%
%   Arguments:
%       input_file   — input CSV path
%       output_file  — output CSV path
%       cols_to_keep — cell array of column names to keep
%       new_names    — cell array of new names (same order)

    if numel(cols_to_keep) ~= numel(new_names)
        error('filter_and_rename_csv: cols_to_keep and new_names must have the same length.');
    end

    %% Read CSV
    T = readtable(input_file);

    available = T.Properties.VariableNames;

    %% Check requested columns
    missing = setdiff(cols_to_keep, available);
    if ~isempty(missing)
        warning('filter_and_rename_csv: columns not found in CSV: %s', ...
            strjoin(missing, ', '));
        % drop the missing ones from the request list
        keep_mask = ismember(cols_to_keep, available);
        cols_to_keep = cols_to_keep(keep_mask);
        new_names    = new_names(keep_mask);
    end

    %% Select and rename
    T_out = T(:, cols_to_keep);
    T_out.Properties.VariableNames = new_names;

    %% Ensure the destination folder exists
    out_dir = fileparts(output_file);
    if ~isempty(out_dir) && ~isfolder(out_dir)
        mkdir(out_dir);
    end

    %% Write
    writetable(T_out, output_file);

end
