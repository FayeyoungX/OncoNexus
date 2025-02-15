#!/python

import os,re,sys,glob
import pandas as pd

block = sys.argv[1]
block_dic = {}
block_df = pd.read_csv(block, sep = "\t")
for Ensemble_ID, Gene_Symbol in zip(block_df['Ensemble.ID'], block_df['Gene.Symbol']):
	block_dic.setdefault(Ensemble_ID, Gene_Symbol)

block_ID_dic = {}
for Ensemble_ID, Symble_ID in zip(block_df['Ensemble.ID'], block_df['Symble.ID']):
	block_ID_dic.setdefault(Ensemble_ID, Symble_ID)

DESeq_indir = sys.argv[2]
for pardir, subdirs, curfiles in os.walk(DESeq_indir):
	for subfile in curfiles:
		if re.search(r"DESeq2.results.xls", subfile):
			tumor = subfile.split(".")[0]
			DESeq_abspath = pardir + "/" + subfile
			DESeq_df = pd.read_csv(DESeq_abspath, sep = "\t", index_col = "Gene")
			Symble_ID = []
			for Ensemble_ID in DESeq_df.index:
				Symble_ID.append(block_ID_dic.get(Ensemble_ID))

			DESeq_df = DESeq_df.rename(block_dic, axis='index')
			DESeq_df['Symbol.ID'] = Symble_ID
			outfile = pardir + "/" + tumor + ".Tumor_vs_Normal.DESeq2.Symbol.results.xls"
			DESeq_df.to_csv(outfile, sep = "\t", index_label = "Gene")


