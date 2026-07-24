# The student release LaTeX files use minted for code highlighting.
# minted needs shell escape so XeLaTeX can invoke Python/Pygments.
$xelatex = 'xelatex -shell-escape %O %S';
$pdf_mode = 5;
