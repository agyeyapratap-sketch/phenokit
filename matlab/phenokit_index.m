function v = phenokit_index(cube, wl, name, tolNm)
%PHENOKIT_INDEX Vegetation index by wavelength (same definitions as phenokit.indices).
%   v = phenokit_index(cube, wl, 'NDVI') with cube H x W x B and wl in nm.
%   Supported: NDVI, GNDVI, NDRE, CIre, SAVI, EVI, PRI, WBI, MCARI.
%   Raises an error if the sensor has no band within tolNm (default 15 nm) of a
%   required wavelength, instead of silently using the wrong band.
if nargin < 4, tolNm = 15; end
b = @(lam) band(cube, wl, lam, tolNm);
nd = @(a, c) (a - c) ./ (a + c + eps);
switch upper(name)
    case 'NDVI',  v = nd(b(800), b(670));
    case 'GNDVI', v = nd(b(800), b(550));
    case 'NDRE',  v = nd(b(790), b(720));
    case 'CIRE',  v = b(790) ./ (b(720) + eps) - 1;
    case 'SAVI',  n = b(800); r = b(670); v = 1.5 * (n - r) ./ (n + r + 0.5);
    case 'EVI',   n = b(800); r = b(670); bl = b(470); v = 2.5 * (n - r) ./ (n + 6*r - 7.5*bl + 1);
    case 'PRI',   v = nd(b(531), b(570));
    case 'WBI',   v = b(900) ./ (b(970) + eps);
    case 'MCARI', r7 = b(700); r6 = b(670); r5 = b(550); v = ((r7 - r6) - 0.2*(r7 - r5)) .* (r7 ./ (r6 + eps));
    otherwise, error('phenokit:index', 'Unknown index %s', name);
end
end

function x = band(cube, wl, lam, tol)
[d, i] = min(abs(wl - lam));
if d > tol
    error('phenokit:index', 'No band within %g nm of %g nm (nearest %g nm)', tol, lam, wl(i));
end
x = cube(:, :, i);
end
