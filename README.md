<p align="center"><img src="./assets/banner.svg" alt="Excelerate EDA: sign-ups and program applications on an online platform" width="100%"></p>

Exploratory data analysis from my data analyst internship at **Excelerate**: data from an online platform where users sign up and apply for programs, covering demographics and program applications.

**[View the Looker Studio dashboard](https://lookerstudio.google.com/reporting/188e6859-77d7-4b1e-8ef8-ef2f788fe08a)**

## What the analysis covers

The script `Execelerate Revised Opportunity Wise.py` (exported from a Jupyter notebook) works through the "Revised Opportunity Wise" dataset with **pandas** and **matplotlib**:

- counts of unique users and opportunities
- which sponsor combinations appear most often
- gender breakdown of applicants (bar and pie charts)
- sign-ups by year
- distributions, plotted as histograms

The findings were turned into the Looker Studio dashboard above and presented to technical and non-technical audiences.

## Run it

```bash
pip install pandas matplotlib
python "Execelerate Revised Opportunity Wise.py"
```

The script reads the dataset from a local CSV path. Change the path at the top to where your copy is.

---

<sub>Part of <b>the silent data</b> · Kennedy Njoroge</sub>
