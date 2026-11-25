function report = phenokit_verify_manifest(rootDir)
%PHENOKIT_VERIFY_MANIFEST Verify files against a phenokit MANIFEST.sha256.csv (SHA-256).
%   report = phenokit_verify_manifest('data/synthetic/SYN-WA-2026') returns a
%   struct with fields ok, missing, modified. Uses Java's MessageDigest, so it
%   runs in base MATLAB without toolboxes.
T = readtable(fullfile(rootDir, 'MANIFEST.sha256.csv'), 'TextType', 'string', 'Delimiter', ',');
missing = strings(0); modified = strings(0);
md = java.security.MessageDigest.getInstance('SHA-256');
for k = 1:height(T)
    f = fullfile(rootDir, T.path(k));
    if ~isfile(f), missing(end+1) = T.path(k); continue; end %#ok<AGROW>
    fid = fopen(f, 'r'); bytes = fread(fid, Inf, '*uint8'); fclose(fid);
    md.reset(); md.update(bytes);
    h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2)', 1, []));
    if ~strcmp(h, T.sha256(k)), modified(end+1) = T.path(k); end %#ok<AGROW>
end
report = struct('ok', isempty(missing) && isempty(modified), 'missing', missing, 'modified', modified);
end
