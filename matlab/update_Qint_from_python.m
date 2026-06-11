function update_Qint_from_python(action_file)
% UPDATE_QINT_FROM_PYTHON
%   Read Qint from the action.csv written by Python and apply it to the
%   Qintr variable in the MATLAB workspace.
%
%   Naming note: the CSV column is named 'Qec' for historical
%   compatibility of the shared communication file. The value is applied
%   as Qintr (internal recirculation), NOT as external carbon. The
%   distinction is made in RL_main_simple.m via the AGENT variable, which
%   chooses which update_* function is called.
%
%   BSM2 limits (Table 18): Qint in [0, 309720] m3/d.
%   Practical limit used in training: [5000, 61944] m3/d
%   (61944 = BSM2 initialization default, Section 9).

    QINT_MIN = 5000.0;
    QINT_MAX = 61944.0;

    try
        data = readmatrix(action_file);
        data = data(~isnan(data));
        Qintr = data(end);   % 'Qec' column = Qint by historical convention
        Qintr = max(QINT_MIN, min(QINT_MAX, Qintr));
    catch e
        warning('update_Qint_from_python: %s — using BSM2 default (61944 m3/d)', e.message);
        Qintr = QINT_MAX;
    end

    assignin('base', 'Qintr', Qintr);
    fprintf('[RL] Qint = %.1f m3/d\n', Qintr);

end
