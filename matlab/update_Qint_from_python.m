function update_Qint_from_python(action_file)
% UPDATE_QINT_FROM_PYTHON
%   Lê Qint do action.csv escrito pelo Python e aplica no workspace MATLAB.
%
%   Nota de nomenclatura: a coluna no CSV chama-se 'Qec' por
%   compatibilidade histórica, mas o valor é aplicado como Qintr
%   (recirculação interna), não como carbono externo.
%
%   Limites BSM2: Qint in [5000, 61944] m³/d

    QINT_MIN = 5000.0;
    QINT_MAX = 61944.0;

    try
        T     = readtable(action_file);
        Qintr = T.Qec(end);   % coluna 'Qec' = Qint por convenção histórica
        Qintr = max(QINT_MIN, min(QINT_MAX, Qintr));
    catch e
        warning('update_Qint_from_python: %s — usando default', e.message);
        Qintr = QINT_MAX;
    end

    assignin('base', 'Qintr', Qintr);
    fprintf('[RL] Qint = %.1f m3/d\n', Qintr);

end