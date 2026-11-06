# Interview notes: Bank Customer Churn Prediction

## The 2-minute story

> "I worked on a bank dataset of 10,000 customers, where about 20% had closed their accounts. The goal was to predict who leaves next, so the bank can act first.
>
> The data showed churn was double in Germany, peaked at 56% for ages 51 to 60, and was twice as high for inactive members. Customers with 2 products almost never left, but nearly everyone with 3 or 4 did.
>
> I compared Logistic Regression, Random Forest and Gradient Boosting, using balanced class weights because churners are only 20%. Then I tuned the two tree models with a randomized search and 5-fold cross-validation. Tuned Gradient Boosting won, with a ROC-AUC of 0.87, and I picked it by its cross-validation score so the test set stayed a fair check.
>
> Then I made it a business tool. Instead of the default 0.5 cut-off, I picked the threshold that maximises profit, given the cost of a call, the value of a saved customer and a save rate. Model targeting made about 2.9 times more profit than calling everyone, and calling just the riskiest 20% reached 62% of churners. Finally I built a two-page Streamlit app. A user uploads a customer file in any common format (CSV, Excel, PDF or Word). The app matches the columns, cleans messy values and shows a data-quality score, then scores everyone and lists the at-risk customers by name with the reason and a suggested action."

## Questions you'll likely get

**Why not just use accuracy?**
Only 20% churn. A model that predicts "nobody leaves" is 80% accurate and useless. ROC-AUC measures ranking quality, and recall and precision show how many churners we catch and how many false alarms we raise.

**What does `class_weight="balanced"` do?**
It makes mistakes on the smaller class (churners) count more in training, so the model doesn't ignore them. A side effect is that predicted probabilities shift upwards, which is part of why the best threshold sits near 0.5 in the base case.

**Why tune with cross-validation and choose by CV score?**
If you pick settings or models by test score, you slowly fit to the test set, and your reported score becomes too optimistic. CV uses only training data, which keeps the test set honest.

**How did you choose the threshold?**
For each threshold I calculated: churners caught × save rate × customer value − customers called × call cost. Then I picked the maximum. With an expensive offer the best threshold rises to 0.77 (call fewer people). With a cheap SMS it drops to 0.18 (call many more).

**Why did tree models beat Logistic Regression?**
Churn isn't a straight line. It rises with age then falls after 60, and it's high for 1 product, low for 2 and very high for 3-4. Trees capture those bends.

**What is permutation importance?**
It shuffles one column at a time and measures how much the test score drops. It reflects real predictive value better than a tree's built-in importance.

**How would this run in a real bank?**
Score all customers monthly with the saved model, send everyone above the threshold to the retention team's CRM, track who was saved, and retrain every quarter.

**How does the app read PDF and Word files?**
It pulls tables out with `pdfplumber` (PDF) and `python-docx` (Word), then treats them like a spreadsheet. Column names are matched using a list of common alternatives, and the user can correct any match. If a PDF has no real table (for example a scanned image), the app says so instead of crashing.

**What happens with messy data, like text in number columns?**
It cleans the data first, in a separate module (`cleaner.py`), and never silently drops a customer. For each column, it:
1. **Reads what it can.** It strips currency symbols and commas, understands `50k`, `1.2 lakh` and `2 cr`, reads `45 yrs` as 45 and `forty` as 40, and fixes a letter O typed for zero (`7OO`).
2. **Fixes typos in categories.** It uses an edit-distance check that counts a swap of two letters as one edit, so `frnace` becomes France and `Germny` becomes Germany. It only corrects a word when every close match means the same thing, so `Finland` is *not* turned into France.
3. **Catches impossible values.** An age of 250 or a credit score of 5000 is treated as a mistake. Obvious ones are fixed: a birth year typed as an age becomes the age, a negative balance becomes 0, and 7 products is capped at 4, the most the model knows.
4. **Fills what's left** (blanks, `N/A`, unreadable text) with the typical value in the file itself (the median for numbers, the most common value for categories), and flags that customer in a "Data check" column. If a column is missing or nearly empty, it uses the training data's typical customer instead.

It also removes duplicate rows, repeated customer IDs and empty rows, and finds the real header row when a file has a title above it. A safety rule stops it from de-duplicating on a column that isn't really unique: if over 20% of rows share an ID, it leaves them alone.

Everything is counted in a cleaning report. For example, my test file had 211 rows: it removed 10 duplicates and 3 empty rows, fixed 92 messy values and guessed 87. The quality score drops as more important values are guessed, so the user knows how far to trust the results.

The order of the row checks matters. I remove empty rows *before* checking for repeated customer IDs. Otherwise, if a customer's first row was empty and a later row was good, the good row would be thrown away as a "duplicate" and the empty one would be removed too, so the customer would vanish. I found that bug by checking the review screen row by row.

**How does the user see what was wrong?**
As soon as the file is read, a pop-up says, for example, "67 of 211 rows have missing or wrong values in 10 columns". One click opens the whole file with the problem cells highlighted: yellow for missing or wrong (a typical value will be used), green for fixed automatically, and red for rows that will be removed, with notes like "Exact copy of row 16". Each row has a plain-English "What's wrong" note, and every problem row can be downloaded to fix the source data.

**How does that stay fast on a million rows?**
The cleaner stores one small status code per cell (a number from 0 to 6) instead of text, so finding problem rows is quick NumPy work. Only the page being looked at (500 rows) gets colours and notes. The notes for the download are built a column at a time, not row by row, which cut that from 22 seconds to about 6 for a million rows. The download is only built when the button is clicked.

**Why fill with typical values instead of dropping the rows?**
Dropping is worse for a retention team, because a customer with one bad cell would silently disappear from the call list. My first version did that and lost 28% of a messy file. With filling, every customer is scored, the guess is visible, and someone can correct it on the single-customer page.

**What if a file is missing whole columns?**
It still scores. Missing inputs are filled with the typical value, and the quality score falls. The score is the share of the model's information the file really provides, weighted by permutation importance, so a file with only age and country gives 43% ("Limited"). If no columns match at all, the app refuses to score and explains why.

**What if I give you a completely different dataset, like an Indian bank's?**
"Try it." The app is built for that:
- **Column names** are matched by their words, not exact spelling. `CIBIL_Score`, `Account_Balance (INR)`, `Num_Products`, `DOB`, `State` and `Months_on_book` all get matched. Traps like `Months_Inactive_12_mon` (which isn't an active flag) are avoided, and every match can be changed by hand.
- **Indian formats** are read correctly: `₹9,17,700`, `1.2 Cr`, `6.5 LPA`, `6-8 LPA` (midpoint), dates of birth like `12/05/1990`, joining years, and tenure in months.
- **Currency** is detected (₹, $ or €) and converted to euros only for scoring, because the model learned in euros. People still see rupees.
- **Indian states** are shown in blue as "not known to the model", not as errors. The app clearly says the model learned from France, Germany and Spain.

The honest part: the app *works* on Indian data, but the *model* was trained on European customers. The behaviour patterns (age, number of products, activity) probably carry over, but the exact numbers won't. For real use I'd retrain the same pipeline on the bank's own data, adding Indian features like UPI activity, whether it's a salary account, CIBIL score, metro vs rural branch, and FDs or loans held. I can't do that with public data, because Indian banks can't share customer-level data under the DPDP Act 2023.

**How would you know if the model works on my data?**
If the file has a column saying who actually left (`Exited`, `Churn`, `Attrition_Flag`, `Account Closed`…), the app finds it and shows an accuracy check: AUC, the share of leavers it caught, the share of flagged customers who really left, and how many leavers the riskiest 20% reached. Two real examples:
- The original bank file: AUC 0.88. The app warns that this is optimistic, because it's the training data. The fair number is my test set's 0.869.
- IBM's public telecom churn data: only 2 of 10 inputs exist, so quality shows "Very limited (3%)" and AUC is 0.58. That's honest: it shows what happens when a model meets data it wasn't built for.

**Why convert rupees to euros?**
For unit consistency. The model learned "balance = 100,000" in euros, so ₹1,00,000 must be converted (to about €940) before it means the same thing to the model. Converting the currency doesn't make Indian customers behave like European ones, though. That's why the app also says to retrain for a new market.

**How did you make sure it never crashes?**
Besides normal tests, I wrote a fuzz test that throws about 1,000 randomly broken files at the app: random column names (repeated, empty, emoji), text in number columns, infinity, impossible dates, empty files, damaged Excel. It found two real bugs that I then fixed:
- A huge number multiplied by a unit ("LPA") overflowed to infinity, and the model refused it. Absurd amounts are now treated as impossible, and there's a final check before scoring.
- A damaged Excel file gave a technical XML error. It now gives a plain-English reason.

**Will it work on a big file?**
Yes. I tested it with 1,000,000 customers, both clean and messy. Each takes about 20-30 seconds from upload to results. To keep it fast, cleaning works once per *distinct* value (a million rows might have only a few hundred different spellings), and risk levels and reasons are calculated for the whole table at once with NumPy instead of row by row. The page shows only the riskiest 1,000 rows, and the download file is built once and cached. The upload limit is 1 GB.

**What is the what-if panel?**
It re-scores the same customer with one thing changed, for example "active member = yes" or "2 products". The bank can then see which action would lower the risk the most. It's a simple form of counterfactual analysis.

**Limitations**
- It's a public sample dataset with no dates or behaviour over time (logins, transactions)
- The cost numbers are assumptions; the code makes them easy to replace
- The patterns are correlations, not proof of cause

## Final model at the 0.5 threshold (2,000 test customers)

| | Predicted stay | Predicted leave |
|---|---|---|
| **Actually stayed** | 1,282 | 311 (false alarms) |
| **Actually left** | 92 (missed) | 315 (caught) |

Recall = 77%, precision = 50%.

## Resume bullet

> Built an end-to-end bank churn model on 10,000 customers (Python, scikit-learn; tuned Gradient Boosting, ROC-AUC 0.87) with a profit-optimised call threshold and a two-page Streamlit app (live online) that cleans and scores any uploaded customer file (CSV/Excel/PDF/Word, Indian ₹/LPA/CIBIL formats, automatic column matching and currency conversion), checks its own accuracy when the file includes real outcomes, and lists at-risk customers by name. Targeting the riskiest 20% reaches 62% of churners, and model targeting earns 2.9× more than calling everyone.
