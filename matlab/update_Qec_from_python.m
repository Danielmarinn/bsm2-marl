function update_Qec_from_python(action_file)
% UPDATE_QEC_FROM_PYTHON
%   Read the external carbon flow (Qec) from the action.csv written by
%   Python and apply it to the carb1 variable in the MATLAB workspace.
%
%   carb1 is the standard BSM2 variable for external carbon dosing in
%   reactor 1 (anoxic) — the dosing point for improving denitrification.
%   Confirmed via `whos carb*` after init_bsm2: scalar 1x1 double.
%
%   BSM2 limits (Table 18, "Available control handles"):
%       Qec in [0, 5] m3/d
%
%   External carbon concentration: CARBONSOURCECONC = 400000 g COD/m3
%   (constant in the workspace — not changed by this agent).
%
%   BSM2 default (Section 9, "Initialization"): Qec = 2 m3/d.

    QEC_MIN = 0.0;
    QEC_MAX = 5.0;

    try
        data = readmatrix(action_file);
        data = data(~isnan(data));
        Qec = data(end);
        Qec = max(QEC_MIN, min(QEC_MAX, Qec));
    catch e
        warning('update_Qec_from_python: %s — using BSM2 default (2 m3/d)', e.message);
        Qec = 2.0;
    end

    % Apply to reactor 1 (anoxic dosing point)
    assignin('base', 'carb1', Qec);

    fprintf('[RL] Qec = %.3f m3/d (carb1)\n', Qec);

end
