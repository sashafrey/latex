$filenames = @("FreiAutoref.tex","chapter1.tex","chapter2.tex","chapter3.tex","chapter4.tex","chapter5.tex","intro.tex","final.tex")
foreach ($filename in $filenames) {
    $mustHavePrefix = @("`$A`$", "`$a`$", "`$x`$", "`$\XX`$", "`$X`$", "\ref{", "\eqref{", "\cite{", "`$L`$", "`$s`$", "`$m`$", "$`A(X)`$");
    $mustHaveSuffix = @("в", "и", "а", "к", "с", "у", "на", "из", "не", "от", "по", "то")

    $sb = New-Object -TypeName "System.Text.StringBuilder"
    $content = Get-Content $filename 
    for ($i = 0; $i -lt $content.Length; $i = $i + 1) {
	    foreach($pref in $mustHavePrefix) {$content[$i] = $content[$i].Replace(" $pref", "~$pref") }
	    foreach($suff in $mustHaveSuffix) 
        {
            $content[$i] = $content[$i].Replace(" $suff ", " $suff~") ;
            if ($content[$i].StartsWith("$suff ")) {
                [void]$sb.Clear();
                [void]$sb.Append($content[$i]);
                $sb[$suff.Length] = '~';
                $content[$i] = $sb.ToString();
            }
        }
    }

    Set-Content "$filename.new" $content
    C:\bin\windiff.exe $filename "$filename.new"
}