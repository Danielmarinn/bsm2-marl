function save_last_sample_to_csv(ws_var, output_file)
% SAVE_LAST_SAMPLE_TO_CSV
%   Extrai a última amostra de uma variável To Workspace (timeseries ou struct)
%   e guarda em CSV.
%
%   Uso:
%       save_last_sample_to_csv(A_RB_in1, 'C:/caminho/raw_state.csv')
%
%   Formatos suportados do bloco To Workspace:
%       - Structure with time  (SaveFormat = 'Structure With Time')
%       - Timeseries           (SaveFormat = 'Timeseries')
%       - Array                (SaveFormat = 'Array')

    %% --- Structure With Time ---
    if isstruct(ws_var) && isfield(ws_var, 'signals') && isfield(ws_var, 'time')

        signals = ws_var.signals;
        t       = ws_var.time;

        % construir tabela com todas as colunas de sinais
        T = table();
        T.time = t(end);   % só última amostra

        if isstruct(signals)
            % pode ser array de structs (um por sinal)
            for k = 1:numel(signals)
                sig    = signals(k);
                vals   = sig.values;
                label  = sig.label;

                % última amostra (última linha)
                last = vals(end, :);

                if size(last, 2) == 1
                    T.(label) = last;
                else
                    % sinal vectorial — criar col_1, col_2, ...
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

    %% --- Array simples (linhas = amostras, colunas = sinais) ---
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

    error('save_last_sample_to_csv: formato não suportado — %s', class(ws_var));
end
