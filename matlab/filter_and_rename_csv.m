function filter_and_rename_csv(input_file, output_file, cols_to_keep, new_names)
% FILTER_AND_RENAME_CSV
%   Lê um CSV, seleciona colunas específicas, renomeia-as e guarda novo CSV.
%
%   Uso:
%       filter_and_rename_csv( ...
%           'raw_state.csv', ...
%           'state.csv', ...
%           {'S_NO', 'S_NH'}, ...
%           {'SNO_anox', 'SNH_in'})
%
%   Argumentos:
%       input_file   — caminho do CSV de entrada
%       output_file  — caminho do CSV de saída
%       cols_to_keep — cell array com nomes das colunas a manter
%       new_names    — cell array com novos nomes (mesma ordem)

    if numel(cols_to_keep) ~= numel(new_names)
        error('filter_and_rename_csv: cols_to_keep e new_names têm de ter o mesmo tamanho.');
    end

    %% Ler CSV
    T = readtable(input_file);

    available = T.Properties.VariableNames;

    %% Verificar colunas pedidas
    missing = setdiff(cols_to_keep, available);
    if ~isempty(missing)
        warning('filter_and_rename_csv: colunas não encontradas no CSV: %s', ...
            strjoin(missing, ', '));
        % remover as que faltam da lista de pedido
        keep_mask = ismember(cols_to_keep, available);
        cols_to_keep = cols_to_keep(keep_mask);
        new_names    = new_names(keep_mask);
    end

    %% Selecionar e renomear
    T_out = T(:, cols_to_keep);
    T_out.Properties.VariableNames = new_names;

    %% Garantir que a pasta de destino existe
    out_dir = fileparts(output_file);
    if ~isempty(out_dir) && ~isfolder(out_dir)
        mkdir(out_dir);
    end

    %% Escrever
    writetable(T_out, output_file);

end
