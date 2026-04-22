function update_Qint_from_python(action_file)
% UPDATE_QINT_FROM_PYTHON
%   Read Qint from action.csv written by Python and apply it in MATLAB.
%
%   Accepted column names:
%     - 'Qint' as the canonical public-facing name
%     - 'Qec' as a legacy compatibility name
%
%   BSM2 limits: Qint in [5000, 61944] m3/d

    QINT_MIN = 5000.0;
    QINT_MAX = 61944.0;

    try
        T = readtable(action_file);

        if ismember('Qint', T.Properties.VariableNames)
            Qintr = T.Qint(end);
        elseif ismember('Qec', T.Properties.VariableNames)
            Qintr = T.Qec(end);
        else
            error('action.csv must contain either Qint or Qec');
        end

        Qintr = max(QINT_MIN, min(QINT_MAX, Qintr));
    catch e
        warning('update_Qint_from_python: %s - using default', e.message);
        Qintr = QINT_MAX;
    end

    assignin('base', 'Qintr', Qintr);
    fprintf('[RL] Qint = %.1f m3/d\n', Qintr);

end
