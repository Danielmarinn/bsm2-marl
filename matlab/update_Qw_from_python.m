function update_Qw_from_python(action_file)
% UPDATE_QW_FROM_PYTHON
%   Read the sludge wastage flow (Qw) from the action.csv written by
%   Python and apply it to the timer-based BSM2 loop.
%
%   To force the Qw_time_controller block to follow the RL action instead
%   of the default low/high alternation, this function writes the same
%   value to Qw, Qw_low and Qw_high in the MATLAB workspace.
%
%   Limits of this CTRL-4 scaffold:
%       Qw in [0, 450] m3/d
%
%   Note: the physical BSM2 actuator allows values above this, but the
%   [0, 450] range keeps the agent within the default timer-based regime
%   used in this thesis.

    QW_MIN = 0.0;
    QW_MAX = 450.0;

    try
        data = readmatrix(action_file);
        data = data(~isnan(data));
        Qw = data(end);
        Qw = max(QW_MIN, min(QW_MAX, Qw));
    catch e
        warning('update_Qw_from_python: %s - using BSM2 default (300 m3/d)', e.message);
        Qw = 300.0;
    end

    assignin('base', 'Qw', Qw);
    assignin('base', 'Qw_low', Qw);
    assignin('base', 'Qw_high', Qw);

    fprintf('[RL] Qw = %.3f m3/d (Qw, Qw_low, Qw_high)\n', Qw);

end
