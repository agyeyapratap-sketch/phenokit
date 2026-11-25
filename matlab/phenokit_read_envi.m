function [cube, wl, info] = phenokit_read_envi(hdrPath)
%PHENOKIT_READ_ENVI Read an ENVI image written by phenokit (or most ENVI tools).
%   [cube, wl, info] = phenokit_read_envi('vnir_cube.hdr') returns cube as
%   lines x samples x bands (double), band-centre wavelengths (nm) and the
%   parsed header. Supports bsq/bil/bip, little/big endian, ENVI data types
%   1,2,3,4,5,12. Mirrors phenokit.io.read_envi so MATLAB users can work on
%   the same calibrated products as the Python pipeline.
%
%   Example:
%     [R, wl] = phenokit_read_envi('outputs/synthetic_run/hs_reflectance.hdr');
%     ndvi = phenokit_index(R, wl, 'NDVI');  imagesc(ndvi); axis image; colorbar
txt = fileread(hdrPath);
if ~startsWith(strtrim(txt), 'ENVI')
    error('phenokit:envi', '%s is not an ENVI header', hdrPath);
end
info = struct();
tok = regexp(txt, '(?m)^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)', 'tokens');
for k = 1:numel(tok)
    key = matlab.lang.makeValidName(lower(strtrim(tok{k}{1})));
    info.(key) = strtrim(tok{k}{2});
end
H = str2double(info.lines); W = str2double(info.samples); B = str2double(info.bands);
dt = str2double(info.data_type);
types = containers.Map({1,2,3,4,5,12}, {'uint8','int16','int32','single','double','uint16'});
prec = types(dt);
order = 'ieee-le';
if isfield(info, 'byte_order') && str2double(info.byte_order) == 1, order = 'ieee-be'; end
offset = 0;
if isfield(info, 'header_offset'), offset = str2double(info.header_offset); end
[p, n] = fileparts(hdrPath);
cands = {fullfile(p, n), fullfile(p, [n '.raw']), fullfile(p, [n '.img']), fullfile(p, [n '.dat'])};
dataFile = '';
for k = 1:numel(cands), if isfile(cands{k}), dataFile = cands{k}; break; end, end
if isempty(dataFile), error('phenokit:envi', 'No data file next to %s', hdrPath); end
fid = fopen(dataFile, 'r', order); fseek(fid, offset, 'bof');
raw = fread(fid, H * W * B, ['*' prec]); fclose(fid);
if numel(raw) < H * W * B, error('phenokit:envi', 'Data file truncated'); end
il = lower(strtrim(info.interleave));
switch il
    case 'bsq', cube = permute(reshape(raw, [W, H, B]), [2 1 3]);
    case 'bil', cube = permute(reshape(raw, [W, B, H]), [3 1 2]);
    case 'bip', cube = permute(reshape(raw, [B, W, H]), [3 2 1]);
    otherwise, error('phenokit:envi', 'Unknown interleave %s', il);
end
cube = double(cube);
wl = [];
if isfield(info, 'wavelength')
    wl = str2double(strsplit(regexprep(info.wavelength, '[{}]', ''), ','));
end
end
