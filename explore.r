# Explore issues in the data

library(data.table)
library(Boruta)
library (caret)
library (car)

datadir = '../../Data/Kaggle/airbnb-recruiting-new-user-bookings'
tmpdir = '/tmp'

# Read the data and preprocess
# ==================================================================================================

if (1) {
  # Users
  colClasses = c('character', 'Date', 'character', 'Date', 'factor', 'numeric', 'factor', 'numeric', 'factor', 'factor', 'factor', 'factor', 'factor', 'factor', 'factor', 'factor')
  
  dtr = fread(paste0(datadir, '/train_users_2.csv'), colClasses = colClasses      )
  dte = fread(paste0(datadir, '/test_users.csv'   ), colClasses = colClasses[1:15])
  
  # Merge train and test so that processing is identical
  dte[, country_destination := NA]
  dte[, is.test := T]
  dtr[, is.test := F]
  dat = rbind(dtr, dte)
  
  # Date stuff
  dat[, date_account_created := as.Date(date_account_created)]
  dat[, dac.year  := as.integer(format(date_account_created, '%y'))]
  dat[, dac.month := as.integer(format(date_account_created, '%m'))]
  dat[, dac.day   := as.integer(format(date_account_created, '%d'))]
  dat[, dac.wday  := wday(date_account_created)]
  
  dat[, timestamp_first_active.d := as.Date(timestamp_first_active, format = '%Y%m%d')]
  dat[, timestamp_first_active.t := as.integer(substr(timestamp_first_active, 9, 10))]
  dat[, tfa.year  := as.integer(format(timestamp_first_active.d, '%y'))]
  dat[, tfa.month := as.integer(format(timestamp_first_active.d, '%m'))]
  dat[, tfa.day   := as.integer(format(timestamp_first_active.d, '%d'))]
  dat[, tfa.wday  := wday(timestamp_first_active.d)]
  
  dat[, time.to.first.use := as.integer(timestamp_first_active.d - date_account_created)]
  
  dat[date_first_booking == '', date_first_booking := NA]
  dat[, date_first_booking := as.Date(date_first_booking)]
  
  dat[, time.to.book := as.numeric(date_first_booking - timestamp_first_active.d)]
  
  # Sessions
  dse = fread(paste0(datadir, '/sessions.csv'))
  
  dse = dse[-which(user_id == '')] # ?!
  dse[action_type == '', action_type := action]
  dse[action_detail == '', action_detail := action_type]
  dse[action == '-unknown-', action := '_unknown']
  dse[action_type == '-unknown-', action_type := '_unknown']
  dse[action_detail == '-unknown-', action_detail := '_unknown']
  dse[, act := paste(action, action_type, action_detail, sep = '.')]
  dse[, act2 := shift(act, 1, type = 'lag', fill = 'first'), by = user_id]
  dse[, act.pair := paste(act2, act, sep = '...')]
  
  # It's possible that session activity is largly dictated by device type
  dse[, device.group := 'unknown']
  dse[device_type %in% c('Mac Desktop', 'Windows Desktop', 'Linux Desktop'), device.group := 'Web']
  dse[device_type %in% c('iPhone', 'iPad Tablet'), device.group := 'iOS']
  dse[device_type %in% c('Android Phone', 'Android App Unknown Phone/Tablet'), device.group := 'Android']
  dse[, device.group := as.factor(device.group)]
  
  dse[is.na(secs_elapsed), secs_elapsed := 0]
  dse[, cumdays := cumsum(secs_elapsed) / (24 * 3600), by = 'user_id']
  
  if (0) {
    # If we fix the horizon, it makes sense to fix this as well, but only if it truly reflects
    # calendar time.... anyway, worth a try
    dse = dse[cumdays <= 91]
  }

  if (0) {
    # Experiment: which acts are important for predicting did/didn't book

    # Single actions:
    dse.act = unique(dse[, .(user_id, act)]) # NOTE: this will only allow us to check if a user did an action or not, not how many time they did it
    dse.act = merge(dse.act, dat[, .(id, booked = as.integer(country_destination != 'NDF'))], by.x = 'user_id', by.y = 'id')
    dse.act = dse.act[!is.na(booked)]
    act.0 = dse.act[booked == 0, .N, by = .(act)]
    act.1 = dse.act[booked == 1, .N, by = .(act)]
    act.tables = merge(act.0, act.1, by = 'act', all = T)
    act.tables[is.na(act.tables)] = 0
    N0 = sum(dse.act$booked == 0)
    N1 = sum(dse.act$booked == 1)
    act.tables[, p := chisq.test(matrix(c(N0 - N.x, N.x, N1 - N.y, N.y), nrow = 2))$p.value, by = 'act']
    act.tables[, q := p.adjust(p, method = 'BH')]
    act.tables = act.tables[order(q, decreasing = F)]
    View(act.tables) 
    # => so maybe take the top 250 for an additive model (if interactions then maybe we want all of them)
    
    # Pairwise actions:
    dse.act.pair = unique(dse[, .(user_id, act.pair)]) # NOTE: this will only allow us to check if a user did an action or not, not how many time they did it
    dse.act.pair = merge(dse.act.pair, dat[, .(id, booked = as.integer(country_destination != 'NDF'))], by.x = 'user_id', by.y = 'id')
    dse.act.pair = dse.act.pair[!is.na(booked)]
    act.pair.0 = dse.act.pair[booked == 0, .N, by = .(act.pair)]
    act.pair.1 = dse.act.pair[booked == 1, .N, by = .(act.pair)]
    act.pair.tables = merge(act.pair.0, act.pair.1, by = 'act.pair', all = T)
    act.pair.tables[is.na(act.pair.tables)] = 0
    N0 = sum(dse.act.pair$booked == 0)
    N1 = sum(dse.act.pair$booked == 1)
    act.pair.tables[, p := chisq.test(matrix(c(N0 - N.x, N.x, N1 - N.y, N.y), nrow = 2))$p.value, by = 'act.pair']
    act.pair.tables[, q := p.adjust(p, method = 'BH')]
    act.pair.tables = act.pair.tables[order(q, decreasing = F)]
    act.pair.tables = act.pair.tables[N.x + N.y > 100]
    View(act.pair.tables) 
    # => there are so many... we probably want to take the top 1000-3000. It looks like the 1-grams already contain most of the predictive information anyway though...
    
    # Which acts are important for predicting US/not|book
    dse.act = unique(dse[, .(user_id, act)]) # NOTE: this will only allow us to check if a user did an action or not, not how many time they did it
    dse.act = merge(dse.act, dat[country_destination != 'NDF', .(id, booked.us = as.integer(country_destination == 'US'))], by.x = 'user_id', by.y = 'id')
    dse.act = dse.act[!is.na(booked.us)]
    act.0 = dse.act[booked.us == 0, .N, by = .(act)]
    act.1 = dse.act[booked.us == 1, .N, by = .(act)]
    act.tables = merge(act.0, act.1, by = 'act', all = T)
    act.tables[is.na(act.tables)] = 0
    N0 = sum(dse.act$booked == 0)
    N1 = sum(dse.act$booked == 1)
    act.tables[, p := chisq.test(matrix(c(N0 - N.x, N.x, N1 - N.y, N.y), nrow = 2))$p.value, by = 'act']
    act.tables[, q := p.adjust(p, method = 'BH')]
    act.tables = act.tables[order(q, decreasing = F)]
    View(act.tables)
    # => here we probably need only the top 100 or so, and they're pretty weak
  }
    
  ssn.base = dse[, list(
    ssn.total.actions = .N, 
    ssn.unique.actions = length(unique(act)),
    ssn.nr.devices = length(unique(device_type)),
    ssn.total.time = sum(secs_elapsed), 
    ssn.time.q0   = min     (secs_elapsed),
    ssn.time.q25  = quantile(secs_elapsed, 0.25),
    ssn.time.q50  = median  (secs_elapsed),
    ssn.time.q75  = quantile(secs_elapsed, 0.75),
    ssn.time.q100 = max     (secs_elapsed)
  ), by = .(user_id)]
  
  # Counts of acts per user
  ssn.caa = dcast(dse[, .N, by = .(user_id, act)], user_id ~ act)
  ssn.caa[is.na(ssn.caa)] = 0
  names(ssn.caa) = paste0('ssn.caa.', names(ssn.caa))
  names(ssn.caa)[1] = 'user_id'
  
  # Merge user and session data
  dat = merge(dat, ssn.base, by.x = 'id', by.y = 'user_id', all.x = T)
  dat = merge(dat, ssn.caa , by.x = 'id', by.y = 'user_id', all.x = T)
  
  save(dat, file = paste0(tmpdir, '/explore-data.RData'))
} else {
  load(file = paste0(tmpdir, '/explore-data.RData')) # => dat
}

# Feature relevance
# ==================================================================================================

if (0) {
  modeling.forumla = ~ . - id - date_account_created - timestamp_first_active - 
    timestamp_first_active.d - country_destination - is.test - date_first_booking - time.to.book
  
  modeling.dat = predict(dummyVars(modeling.forumla, data = dat[is.test == F]), newdata = dat[is.test == F])
  modeling.dat[is.na(modeling.dat)] = -999 # FIXME does this make sense for Boruta?
  modeling.dat[!(is.finite(modeling.dat))] = 999
  feature.names = colnames(modeling.dat)
  
  # Ok this just takes too long...
  #set.seed(123)
  #brt = Boruta(modeling.dat, dat[is.test == F, as.numeric(country_destination != 'NDF')], pValue = 0.05, mcAdj = F, maxRuns = 16, doTrace = 2)
  #cat('\nBoruta Decision for Each Attribute\n')
  #attStats(brt)
  
  # Manually go over the features marginally
  # We only actually care about binary session features and how they affect the binary response == NDF,
  # so simple chisq tests are enough
  labels = dat[is.test == F, as.numeric(country_destination != 'NDF')]
  zscores = apply(modeling.dat, 2, function(x) summary(glm(labels ~ x, family = binomial))$coefficients[2, 3])
  names(zscores) = feature.names
  data.frame(zscore = zscores[order(abs(zscores), decreasing = T)])
  
  # TODO: pairwise ('2-gram') features
  #dse[, action2 := shift(action, 1, type = 'lag'), by = user_id]
  #dse[, action_type2 := shift(action_type, 1, type = 'lag'), by = user_id]
  #dse[, action_detail2 := shift(action_detail, 1, type = 'lag'), by = user_id]
  #merge with dat to get label != 'NDF' for each user, split to "cases and controls"
  #[if memory allows...] now count (action X action_type X action_detail X action2 X action_type2 X action_detail2) in each group
  #merge the couts from both groups so we have "A11" in the 2x2 table from each combination of actions
  #do a chisq test
}

# Time to book, session sec_elapsed
# ==================================================================================================

if (0) {
  # Do the sessions stop at the first booking? in both train and test?
  
  # NOTE: I'm not sure my interpretation of the session is correct. What is the meaning of 
  # secs_elapsed? Are there no gaps? are ALL actions included in the session file? probably not...
  
  dat = dat[-which(time.to.book > 365)]
  summary(lm(log1p(dat[is.test == F, ssn.total.time]) ~ dat[is.test == F, time.to.book]))
  plot(dat[is.test == F, time.to.book], dat[is.test == F, ssn.total.time / 3600 / 24], pch = '.')
  
  # According to this, the sessions cover a variable amount of time, and if anything, are inversely
  # proportional to the time to book.
  
  # And the session lengths are marginally the same in train/test
  hist(log(dat[is.test == F, ssn.total.time]))
  hist(log(dat[is.test == T, ssn.total.time]))
  
  # Fix the labels assuming the testset only allowed 91 days to book
  dat[time.to.book > 91, country_destination := 'NDF']
  
  # What about some telltale features?
  dat.ws = dat[ssn.total.actions > 0]
  table(dat.ws[is.test == F, country_destination != 'NDF'], dat.ws[is.test == F, `ssn.caa.pay_-unknown-` > 0])
  table(dat.ws[is.test == F, country_destination != 'NDF'], dat.ws[is.test == F, ssn.caa.at_checkpoint_booking_request > 0])
  table(dat.ws[is.test == F, country_destination != 'NDF'], dat.ws[is.test == F, ssn.caa.receipt_view])
  table(dat.ws[is.test == F, country_destination != 'NDF'], dat.ws[is.test == F, `ssn.caa.add_guests_-unknown-` > 0])
  table(dat.ws[is.test == F, country_destination != 'NDF'], dat.ws[is.test == F, ssn.caa.respond_submit])
  table(dat.ws[is.test == F, country_destination != 'NDF'], dat.ws[is.test == F, ssn.caa._message_post > 0])
  table(dat.ws[is.test == F, country_destination != 'NDF'], dat.ws[is.test == F, ssn.caa.impressions_view > 0])
  table(dat.ws[is.test == F, country_destination != 'NDF'], dat.ws[is.test == F, `ssn.caa.agree_terms_check_-unknown-` > 0])
  table(dat.ws[is.test == F, country_destination != 'NDF'], dat.ws[is.test == F, ssn.caa.cancellation_policies_view > 0])
  table(dat.ws[is.test == F, country_destination != 'NDF'], dat.ws[is.test == F, `ssn.caa.email_itinerary_colorbox_-unknown-` > 0])
  table(dat.ws[is.test == F, country_destination != 'NDF'], dat.ws[is.test == F, ssn.caa.this_hosting_reviews_click > 0])

  table(dat.ws[is.test == F, `ssn.caa.pay_-unknown-` > 0]) / sum(!dat.ws$is.test)
  table(dat.ws[is.test == T, `ssn.caa.pay_-unknown-` > 0]) / sum( dat.ws$is.test)

  table(dat.ws[is.test == F, ssn.caa.at_checkpoint_booking_request > 0]) / sum(!dat.ws$is.test)
  table(dat.ws[is.test == T, ssn.caa.at_checkpoint_booking_request > 0]) / sum( dat.ws$is.test)

  table(dat.ws[is.test == F, ssn.caa._message_post > 0]) / sum(!dat.ws$is.test)
  table(dat.ws[is.test == T, ssn.caa._message_post > 0]) / sum( dat.ws$is.test)
}

# Foreign languages speakers
# ==================================================================================================

if (0) {
  dat[, language := as.factor(language)]
  dat[, country_destination := as.factor(country_destination)]
  
  xx = cbind(table(dat[is.test == F, language]), table(dat[is.test == T, language]))[order(table(dat$language), decreasing = T), ]
  colnames(xx) = c('train', 'test')
  print(round(t(t(xx) / colSums(xx)) * 1000) / 1000)
  
  langs = names(sort(table(dat$language), decreasing = T))
  lang.dests = NULL
  for (i in 1:length(langs)) {
    lang.dests = rbind(lang.dests, table(dat[is.test == F & language == langs[i], country_destination]))
  }
  rownames(lang.dests) = langs
  print(round(t(t(lang.dests[, order(colSums(lang.dests), decreasing = T)] / rowSums(lang.dests))) * 1000) / 1000)
}

# Train/test marginal class distribution discrepancy
# ==================================================================================================

if (0) {
  #             NDF       US    other       FR       IT       GB        ES       CA       DE       NL       AU       PT 
  yp.tr = c(0.58350, 0.29220, 0.04730, 0.02350, 0.01330, 0.01090, 0.010540, 0.00670, 0.00500, 0.00360, 0.00250, 0.00100)
  yp.te = c(0.67909, 0.23470, 0.03403, 0.01283, 0.01004, 0.00730,       NA,      NA, 0.00344,      NA,      NA,      NA)
  # I haven't finished polling for the smaller classes yet, so make them up
  yp.te[7:11] = c(0.007, 0.005, 0.004, 0.003, 0.002)
  yp.te[12] = (1 - sum(yp.te[1:11]))
  
  # This is shocking: we know there is a big difference in class distributions between the train
  # and test sets. Since the split is according to time_first_active, I figured this has to be
  # because some users take their time and do their first booking long after their first activity.
  # But the below (when in DNF/booked mode) shows that time has no marginal effect on the
  # probability to book (at least for users with first activities in 2014, for the entire 
  # trainset the situation is different, but I don't think that is interesting since the testset
  # only covers a few more months of 2014). So what could possibly explain the big difference???
  # Could it be that the evaluation of whether/where the user booked took place at the end or 
  # shortly after the end of the testset period? In that case, it could be that many more of the
  # "very recently added" users did not book.
  
  d = dat[is.test == F & timestamp_first_active.d >= '2014-01-01']
  d[, label := as.numeric(country_destination != 'NDF')]
  d[, date_account_created.int := as.integer(date_account_created)]
  d[, timestamp_first_active.d.int := as.integer(timestamp_first_active.d)]
  table(d$label) / nrow(d)
  hist(d$timestamp_first_active.d.int)
  summary(glm(label ~ timestamp_first_active.d.int, d, family = binomial))
  summary(glm(label ~ d$date_account_created.int, d, family = binomial))
  
  # Ok, new theory: this says that the most recent first booking recorded was exactly one year
  # after the trainset ended:
  summary(dbooked$date_first_booking)
  
  # It would make sense to allow exactly the same horizon for all users, i.e., if no booking was
  # made by the user till a year passed from the date they were first active, declare NDF and stop
  # tracking that user. But according to the following, this is not what they did:
  summary(as.numeric(dbooked$date_first_booking - dbooked$timestamp_first_active.d))
  
  # On the other hand, there are only a handful of such examples:
  head(sort(as.numeric(dbooked$date_first_booking - dbooked$timestamp_first_active.d), decreasing = T), 50)
  
  # So maybe those are data artifacts, and indeed users were tracked for one year.
  
  # About the testset we can only guess... if they indeed gave each user a year to book, then it
  # doesn't look like seasonality or drift alone can explain the big discrepancy between the class
  # distributions. My guess is that (maybe by mistake) the last date they evaluated booking status
  # on is 2015-06-30, so some testset users had a little less than a year to book, while others had
  # even less, and this results in the inflated probability of NDF. 
  
  # Let's test this hypothesis:
  d = dat[timestamp_first_active.d >= '2014-04-01' & timestamp_first_active.d < '2014-07-01']
  d[date_first_booking >= '2015-07-01', country_destination := 'NDF']
  cbind(round(sort(table(d$country_destination) / nrow(d), decreasing = T) * 1e5) / 1e5, yp.te)
  # Doesn't pan out at all... 
  
  # But... playing with the cutoff I see that if it was 2014-09-30 (i.e., maybe they cut off
  # test eval at three months) it would explain everything:
  d = dat[timestamp_first_active.d >= '2014-04-01' & timestamp_first_active.d < '2014-07-01']
  d[date_first_booking >= '2014-10-01', country_destination := 'NDF']
  cbind(round(sort(table(d$country_destination) / nrow(d), decreasing = T) * 1e5) / 1e5, yp.te)
  
  # It's not clear if they gave all test users the same 3 month horizon, or if they cut off at
  # fixed date. We tried the latter above, so now try the former:
  d = dat[timestamp_first_active.d >= '2014-04-01' & timestamp_first_active.d < '2014-07-01']
  d[as.numeric(date_first_booking - timestamp_first_active.d) > 91, country_destination := 'NDF']
  cbind(round(sort(table(d$country_destination) / nrow(d), decreasing = T) * 1e5) / 1e5, yp.te)
  
  # If this is true, then the labels in the testset have a very different meaning than those in
  # the trainset. It's true that most bookings happen on the first couple of activity days:
  xx = as.numeric(as.Date(dat$date_first_booking) - dat$timestamp_first_active.d)
  xx = xx[!is.na(xx)]
  hist(xx[xx < 60], 100)
  # but a sizeable number of users take their time:
  summary(xx)
  # Which leads to the discrepancy I believe.
  
  # In any case, if this is true - we could just fix the trainset labels to reflect the same kind
  # of horizon, and this shuold improve our predictions a lot!
  
  # There is still a possible issue with NDFs, since for those users we don't know the "end of 
  # tracking" date, and so it is possible we have more info about these in the trainset (that they
  # didn't book not only for the "fair" horizon, but also longer than that)
}

# Class distribution near the end of the trainset
# ==================================================================================================

if (0) {
  # There doesn't seem to be a big change in the distribution near the end, and it is not monotonic:
  dd = dat[is.test == F & timestamp_first_active.d > '2014-06-29']
  sort(table(dd$country_destination) / nrow(dd), decreasing = T)
  dd = dat[is.test == F & timestamp_first_active.d > '2014-06-28']
  sort(table(dd$country_destination) / nrow(dd), decreasing = T)
  dd = dat[is.test == F & timestamp_first_active.d > '2014-06-27']
  sort(table(dd$country_destination) / nrow(dd), decreasing = T)
  dd = dat[is.test == F & timestamp_first_active.d > '2014-06-20']
  sort(table(dd$country_destination) / nrow(dd), decreasing = T)
  dd = dat[is.test == F & timestamp_first_active.d > '2014-06-01']
  sort(table(dd$country_destination) / nrow(dd), decreasing = T)
  dd = dat[is.test == F & timestamp_first_active.d > '2014-03-01']
  sort(table(dd$country_destination) / nrow(dd), decreasing = T)
  dd = dat[is.test == F & timestamp_first_active.d > '2014-01-01']
  sort(table(dd$country_destination) / nrow(dd), decreasing = T)
}
