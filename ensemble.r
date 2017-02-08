# Ensemble some of my best models

# Current (naive) approach: give each submission a prior probability (if based on the public LB 
# performance, might be overfitting...) and from each submission, give each destination a score
# equal to its rank in for the user in question, then average. We should expect to do better if we
# had the probabilities of each class (but that's not always well defined...)

library(data.table)

submit.tag = 'ens10'
#sb.nrs     = c(9, 13, 14, 16, 17, 28, 31, 37, 41, 43, 47, 48, 50, 51, 52, 53, 54, 55)
#sb.weights = c(3, 1 , 1 , 1 , 1 , 1 , 1 , 5 , 5 , 3 , 5 , 2 , 2 , 2 , 5 , 2 , 1 , 1 )
sb.nrs     = c('ens5', 'ens4', 'ens7', 'ens8', '42', 'ens9', '52')
sb.weights = c(2     , 2     , 1     , 1     , 1   , 1     , 1   )
sb.weights = sb.weights / sum(sb.weights)
slot.weights = 5:1 # rank avg
#slot.weights = c(0.74456448, 0.16414798, 0.03490710, 0.01382137, 0.01083518) # weighted rank avg, using average sorted probs from model 52

countries = 1:12
names(countries) = c('NDF', 'US', 'other', 'FR', 'IT', 'GB', 'ES', 'CA', 'DE', 'NL', 'AU', 'PT')
ncountries = length(countries)

read.submission = function(sb.nr) {
  sb = read.csv(paste0('submission-', sb.nr, '.csv'))
  sb$country.code = -1
  for (i in 1:ncountries) {
    sb$country.code[sb$country == names(countries)[i]] = countries[i]
  }
  idx = seq(1, nrow(sb), by = 5)
  return (data.frame(id = sb[idx, 1], country1 = sb[idx, 3], country2 = sb[idx + 1, 3], country3 = sb[idx + 2, 3], country4 = sb[idx + 3, 3], country5 = sb[idx + 4, 3]))
}

sb = read.submission(sb.nrs[1])
nusers = nrow(sb)

scoreboard = matrix(0, nusers, ncol = ncountries)

for (si in 1:length(sb.nrs)) {
  sb.nr = sb.nrs[si]
  sb = read.submission(sb.nr)
  sb = sb[order(sb$id), ]
  for (i in 1:5) {
    idx = (1:nusers) + (sb[, 1 + i] - 1) * nusers
    scoreboard[idx] = scoreboard[idx] + sb.weights[si] * slot.weights[i]
  }
}

preds = t(apply(scoreboard, 1, order, decreasing = T))[, 1:5]
preds = data.frame(id = sb$id, country = matrix(names(countries)[c(preds)], dim(preds)))
submission = melt(preds, 'id') # NOTE: ignore the warning about attributes
submission = submission[order(submission$id, submission$variable), -2]
names(submission) = c('id', 'country')

write.csv(submission, paste0('submission-', submit.tag, '.csv'), quote = F, row.names = F)
zip(paste0('submission-', submit.tag, '.zip'), paste0('submission-', submit.tag, '.csv'))

ref.submission = read.csv('submission-ens5.csv') # my best submission so far
ref.submission = ref.submission[seq(1, nrow(ref.submission), by = 5), ]
cmpr = merge(preds, ref.submission, by = 'id')
cat('Sanity check: first prediction per user matches my best submission', mean(as.character(cmpr$country) == as.character(cmpr$country.1)), 'of the time\n')

cat('Distribution of prediction slots:\n')
all.pred.class.dist = NULL
for (i in 1:5) {
  pred.class.dist = table(factor(preds[, i + 1], levels = names(countries)))
  all.pred.class.dist = rbind(all.pred.class.dist, pred.class.dist / sum(pred.class.dist))
}
print(round(all.pred.class.dist * 1000) / 1000)
cat('\n')
