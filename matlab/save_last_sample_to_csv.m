function save_last_sample_to_csv(ws_var, output_file)
% SAVE_LAST_SAMPLE_TO_CSV
%   Extracts the last sample of a To Workspace variable (timeseries or struct)
%   and saves it to CSV.
%
%   Usage:
%       save_last_sample_to_csv(A_RB_in1, 'C:/path/raw_state.csv')
%
%   Supported To Workspace block formats:
%       - Structure with time  (SaveFormat = 'Structure With Time')
%       - Timeseries           (SaveFormat = 'Timeseries')
%       - Array                (SaveFormat = 'Array')

    %% --- Structure With Time ---
    if isstruct(ws_var) && isfield(ws_var, 'signals') && isfield(ws_var, 'time')

        signals = ws_var.signals;
        t       = ws_var.time;

        % build table with all signal columns
        T = table();
        T.time = t(end);   % last sample only

        if isstruct(signals)
            % may be an array of structs (one per signal)
            for k = 1:numel(signals)
                sig    = signals(k);
                vals   = sig.values;
                label  = sig.label;

                % last sample (last row)
                last = vals(end, :);

                if size(last, 2) == 1
                    T.(label) = last;
                else
                    % vector signal — create col_1, col_2, ...
                    for c = 1:size(last, 2)
                        col_name = sprintf('%s_%d', label, c);
                        T.(col_name) = last(c);
                    end
                end
            end
        end

        writetable(T, output_file);
        return
    end

    %% --- Timeseries ---
    if isa(ws_var, 'timeseries')

        data = ws_var.Data;
        t    = ws_var.Time;
        last = data(end, :);

        cols = cell(1, size(last, 2));
        for c = 1:numel(cols)
            cols{c} = sprintf('sig_%d', c);
        end

        T      = array2table(last, 'VariableNames', cols);
        T.time = t(end);
        writetable(T, output_file);
        return
    end

    %% --- Simple array (rows = samples, columns = signals) ---
    if isnumeric(ws_var)

        last = ws_var(end, :);
        cols = cell(1, size(last, 2));
        for c = 1:numel(cols)
            cols{c} = sprintf('sig_%d', c);
        end

        T = array2table(last, 'VariableNames', cols);
        writetable(T, output_file);
        return
    end

    error('save_last_sample_to_csv: unsupported format — %s', class(ws_var));
end
