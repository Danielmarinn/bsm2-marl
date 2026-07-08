function update_Qw_from_python(action_file)
% UPDATE_QW_FROM_PYTHON
%   Le o caudal de sludge wastage (Qw) do action.csv escrito pelo
%   Python e aplica-o ao loop timer-based do BSM2.
%
%   Para forcar o bloco Qw_time_controller a seguir a accao do RL em
%   vez da alternancia low/high default, esta funcao escreve o mesmo
%   valor em Qw, Qw_low e Qw_high no workspace MATLAB.
%
%   Limites desta scaffold CTRL-4:
%       Qw in [0, 450] m3/d
%
%   Nota: o actuador fisico BSM2 permite valores acima disto, mas o
%   intervalo [0, 450] mantem o agente dentro do regime timer-based
%   default usado nesta tese.

    QW_MIN = 0.0;
    QW_MAX = 450.0;

    try
        data = readmatrix(action_file);
        data = data(~isnan(data));
        Qw = data(end);
        Qw = max(QW_MIN, min(QW_MAX, Qw));
    catch e
        warning('update_Qw_from_python: %s - usando default BSM2 (300 m3/d)', e.message);
        Qw = 300.0;
    end

    assignin('base', 'Qw', Qw);
    assignin('base', 'Qw_low', Qw);
    assignin('base', 'Qw_high', Qw);

    fprintf('[RL] Qw = %.3f m3/d (Qw, Qw_low, Qw_high)\n', Qw);

end
