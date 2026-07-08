function update_Qint_from_python(action_file)
% UPDATE_QINT_FROM_PYTHON
%   Le Qint do action.csv escrito pelo Python e aplica-o a variavel
%   Qintr do workspace MATLAB.
%
%   Nota de nomenclatura: a coluna no CSV chama-se 'Qec' por razoes
%   historicas de compatibilidade do ficheiro de comunicacao
%   partilhado. O valor e' aplicado como Qintr (recirculacao interna),
%   NAO como carbono externo. A distincao e' feita em RL_main_simple.m
%   via a variavel AGENT, que escolhe qual funcao update_* e' chamada.
%
%   Limites BSM2 (Table 18): Qint in [0, 309720] m3/d.
%   Limite pratico usado no treino: [5000, 61944] m3/d
%   (61944 = valor default da inicializacao BSM2, Section 9).

    QINT_MIN = 5000.0;
    QINT_MAX = 61944.0;

    try
        data = readmatrix(action_file);
        data = data(~isnan(data));
        Qintr = data(end);   % coluna 'Qec' = Qint por convencao historica
        Qintr = max(QINT_MIN, min(QINT_MAX, Qintr));
    catch e
        warning('update_Qint_from_python: %s — usando default BSM2 (61944 m3/d)', e.message);
        Qintr = QINT_MAX;
    end

    assignin('base', 'Qintr', Qintr);
    fprintf('[RL] Qint = %.1f m3/d\n', Qintr);

end
