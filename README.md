# Fraud Detection: A Daily Review Queue for Mobile Money

## 1. Business problem

**Client and context.** The imagined client is a South African mobile money provider (a bank-backed wallet that lets customers send money to each other and cash out at agents and ATMs). Its fraud team reviews suspicious transactions by hand: an analyst looks at the transaction, may call the customer, and can hold or reverse it.

**Question.** Which transactions should the fraud team review each day, so that it catches most fraud without burying the team in false alarms it learns to ignore?

**Cost of mistakes.**

| Mistake | What happens | Assumed cost |
|---|---|---|
| False negative (missed fraud) | The money leaves the customer's account and is cashed out. The provider usually reimburses the customer and loses trust. | The full transaction amount, typically thousands of rand or more, plus the investigation |
| False positive (false alarm) | An analyst spends about 15 minutes on a legitimate transaction; the customer may get a call or a short hold | About R60 of analyst time, plus some customer annoyance |

A missed fraud costs hundreds or thousands of times more than a false alarm. But false alarms have a hidden cost too: if most flags turn out to be nothing, analysts start skimming them, and real fraud slips through a queue nobody trusts.

**Capacity.** I assume a team of **6 analysts, each reviewing about 50 cases a day: 300 reviews a day**. That number, not a probability of 0.5, sets how many transactions the model may flag.

**Metric and reasoning.** Accuracy is useless: fraud is so rare that "nothing is fraud" is over 99% accurate. The measures that matter are:

- **Recall within capacity:** of all fraud, what share lands in the daily top 300?
- **Precision of the queue:** what share of the 300 reviews are real fraud? This decides whether the team trusts the queue.
- **Value caught:** how much money the queue protects, because one large theft matters more than several small ones.

I compare models with the **precision-recall curve and PR AUC**, not ROC AUC alone, because ROC AUC looks flattering on rare events.

## 2. Data

**Source:** [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (CC BY-SA 4.0), a simulated mobile money transaction log built from one month of real transactions at a mobile money service in an African country (Lopez-Rojas, Elmir and Axelsson, 2016, *PaySim: A financial mobile money simulator for fraud detection*). The notebook downloads an identical copy from a [Hugging Face mirror](https://huggingface.co/datasets/purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection) because it needs no login. 6,362,620 transactions over 743 hours (31 days).

Real transaction-level fraud data is almost never public, in South Africa or anywhere else, because of privacy law (POPIA here) and bank secrecy. A simulation calibrated on real data is the standard way to show the method.

| Column | Meaning |
|---|---|
| `step` | Hour of the simulation, 1 to 743. Hour of day and day number are derived from it. |
| `type` | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT` or `TRANSFER` |
| `amount` | Transaction amount in the simulator's currency (read as rand for this framing) |
| `nameOrig` | Account sending the money. Always a customer (`C...`). |
| `oldbalanceOrg`, `newbalanceOrig` | Sender's balance **before** and **after** the transaction |
| `nameDest` | Receiving account: customer (`C...`) or merchant (`M...`, payments only) |
| `oldbalanceDest`, `newbalanceDest` | Receiver's balance before and after. Not recorded for merchants (always 0). |
| `isFraud` | Label: transaction made by one of the simulator's fraudster agents, who take over an account and try to empty it |
| `isFlaggedFraud` | The provider's existing rule, aimed at very large transfers. It fired 16 times in the whole month. |

**Before and after.** A fraud check happens *before* a transaction is approved, so the "after" balances would not be available to a real model. Step 3 checks whether they leak the answer.
