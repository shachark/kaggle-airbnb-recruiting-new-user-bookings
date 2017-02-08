# Kaggle competition "airbnb-recruiting-new-user-bookings"

############## Competition description:
#
# In this challenge, you are given a list of users along with their demographics, web session
# records, and some summary statistics. You are asked to predict which country a new user's first
# booking destination will be. All the users in this dataset are from the USA.
#
# There are 12 possible outcomes of the destination country: 
# 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF' (no destination found), and 'other'.
# Please note that 'NDF' is different from 'other' because 'other' means there was a booking, but
# is to a country not included in the list, while 'NDF' means there wasn't a booking.
#
# The training and test sets are split by dates. In the test set, you will predict all the new users
# with first activities after 7/1/2014 (note: this is updated on 12/5/15 when the competition
# restarted). In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset
# dates back to 2010. 
#
##############

library (data.table)
library (xgboost)
library (caret)
library (car)
library (ComputeBackend)

config = list()

config$data.tag        = '1'
config$model.tag       = '1'
config$submit.tag      = '55'

config$mode            = 'complex' # { single, complex, tuning, bagging }
config$complex.select  = 'three-stage' # for 'complex' mode

config$do.preprocess   = F
config$do.cv           = F
config$do.manual.cv1   = F
config$do.manual.cv2   = F
config$do.train        = F
config$do.postprocess  = F
config$do.submission   = F

config$do.select.validset     = F
config$do.preprocess.users    = T
config$do.preprocess.sessions = F
config$do.compile.data        = T

config$do.importance = F

config$validation.scheme  = 'none' # { none, random, last }
config$validation.frac    = 0.1
config$stratify.validset  = T

config$session.features   = c('base', 'caaa', 'faaa')

# Main approachs I tried: { 'allclass', 'top5', 'NDF/booked', 'NDF/US/rest', 'US/notUS|booked', 'destination|booked', 'destination|bookedNotUS', 'other/FR/IT|oneOfThese' }

# This is where most of the signal is
config$s1.main.approach = 'NDF/booked'
config$s1.nr.classes = 2
config$s1.params$xgb.nrounds           = 1800
config$s1.params$xgb.eta               = 0.01
config$s1.params$xgb.max_depth         = 10
config$s1.params$xgb.min.child.weight  = 10
config$s1.params$xgb.subsample         = 0.8
config$s1.params$xgb.colsample_bytree  = 0.3

# This is already a very difficult problem for which we have few weakly relevant features, and 
# not that many samples
config$s2.main.approach = 'US/notUS|booked'
config$s2.nr.classes = 2
config$s2.params$xgb.nrounds           = 900
config$s2.params$xgb.eta               = 0.01
config$s2.params$xgb.max_depth         = 10
config$s2.params$xgb.min.child.weight  = 10
config$s2.params$xgb.subsample         = 0.8
config$s2.params$xgb.colsample_bytree  = 0.3

# Here it is close to impossible to improve on a constant model. The remaining classes can be
# estimated from the marginal public LB distribution or just as zero so they are never selected
# to the top 5
config$s3.main.approach = 'other/FR/IT|oneOfThese'
config$s3.nr.classes = 3
config$s3.params$xgb.nrounds           = 1900
config$s3.params$xgb.eta               = 0.01
config$s3.params$xgb.max_depth         = 3
config$s3.params$xgb.min.child.weight  = 1
config$s3.params$xgb.subsample         = 0.5
config$s3.params$xgb.colsample_bytree  = 0.3

if (config$mode == 'single') {
  if (config$model.tag == '1') {
    config$main.approach = config$s1.main.approach
    config$xgb.params = config$s1.params
  } else if (config$model.tag == '2') {
    config$main.approach = config$s2.main.approach
    config$xgb.params = config$s2.params
  } else if (config$model.tag == '3') {
    config$main.approach = config$s3.main.approach
    config$xgb.params = config$s3.params
  } else if (config$model.tag == 'debug') {
    config$main.approach = 'NDF/US/rest'
    config$nr.classes = 3
    config$xgb.params$xgb.nrounds           = 2000
    config$xgb.params$xgb.eta               = 0.01
    config$xgb.params$xgb.max_depth         = 5
    config$xgb.params$xgb.min.child.weight  = 1
    config$xgb.params$xgb.subsample         = 0.8
    config$xgb.params$xgb.colsample_bytree  = 0.3
  } else {
    stop('TODO')
  }
  
  config$xgb.params$max.xgb.nrounds       = 5000
  config$xgb.params$xgb.early.stop.round  = ceiling(config$xgb.params$xgb.nrounds / 5)
  config$xgb.params$xgb.print.every.n     = ceiling(config$xgb.params$xgb.nrounds / 20)
}

#
# Experimental stuff
#

config$use.precise.eval   = F
config$weighting.scheme   = 'none' # { none, book, class, time }
config$two.class.thresh   = 0.49 #0.4 #0.36 #0.64
config$naive.expand1      = T
config$naive.expand2      = F
config$naive.expand3      = F
config$naive.expand4      = F
config$careful.expand1    = F
config$careful.expand2    = F
config$careful.expand3    = F
config$careful.expand4    = F
config$post.reweight      = F
config$reweight.lambda    = 1/3 #0.15
config$reweight.v         = c(1.1638218, 0.8032170, 0.7194503, 0.5459574, 0.7548872, 0.6697248, 0.6641366, 0.7462687, 0.8000000, 0.8333333, 0.8000000, 1.0100000) # c(1.2, 0.8, rep(0.7, 10))
config$fix.all            = F
config$fix.va.and.te      = F
config$max.days.to.book   = as.numeric(as.Date('2014-09-30') - as.Date('2014-07-01'))
config$simplify.model     = F
config$experiment.dupfeat = F
config$only.use.2014.data = F
config$post.w             = c(1, 0.91, 0.9, 0.7, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2) #c(1.06, 0.92, rep(1, 10)) #rep(1, 12)
config$three.stage.gamma  = 0.65 #0.37

#
# Problem constants
#

config$nr.classes = (config$main.approach %in% c('NDF/booked', 'US/notUS|booked')) * 2 + (config$main.approach == 'NDF/US/rest') * 3 + (config$main.approach == 'destination|bookedNotUS') * 10 + (config$main.approach == 'destination|booked') * 11 + (config$main.approach == 'top5') * 5 + (config$main.approach == 'allclass') * 12 + (config$main.approach == 'other/FR/IT|oneOfThese') * 3
config$recode.na.as = -999

# Using this throughout for consistency
# NOTE: this is the order of prevalence (apparently, also in the testset)
config$label.coding = c('NDF', 'US', 'other', 'FR', 'IT', 'GB', 'ES', 'CA', 'DE', 'NL', 'AU', 'PT')

# By looking at the trainset, and by polling the public leaderboard, we have the following response
# distributions (the testset one is an estimate for the privae LB, but a pretty good one due to the
# random split there):
#                    NDF       US    other       FR       IT       GB        ES       CA       DE       NL       AU       PT 
config$yp.tr = c(0.58350, 0.29220, 0.04730, 0.02350, 0.01330, 0.01090, 0.010540, 0.00670, 0.00500, 0.00360, 0.00250, 0.00100)
config$yp.te = c(0.67909, 0.23470, 0.03403, 0.01283, 0.01004, 0.00730,       NA,      NA, 0.00344,      NA,      NA,      NA)
# I haven't finished polling for the smaller classes yet, so make them up
config$yp.te[7:11] = c(0.007, 0.005, 0.004, 0.003, 0.002)
config$yp.te[12] = (1 - sum(config$yp.te[1:11]))

# NOTE: with this class distribution, we've got the following baseline scores achieved by trivial
# constant classifiers:
# NDF/booked ................ 0.8535938
# NDF/US/rest ............... 0.8535938
# allclass .................. 0.8535938
# US/notUS|booked ........... 0.8405271
# destination|booked ........ 0.8405271
# destination|bookedNotUS ... 0.6147400

#
# Compute platform stuff
#

config$compute.backend = 'serial' # {serial, multicore, condor, pbs}
config$nr.cores = ifelse(config$compute.backend %in% c('condor', 'pbs'), 200, 3)
config$rng.seed = 123
config$nr.threads = detectCores(all.tests = F, logical = F) # for computation on this machine

config$package.dependencies = c('ComputeBackend', 'data.table', 'xgboost')
config$source.dependencies  = NULL
config$cluster.dependencies = NULL
config$cluster.requirements = 'FreeMemoryMB >= 7000'

config$datadir = '../../Data/Kaggle/airbnb-recruiting-new-user-bookings'
config$project.dir = system('pwd', intern = T)
if (config$compute.backend == 'condor') {
  config$tmp.dir = paste0(config$project.dir, '/tmp')  # need this on the shared FS
} else {
  config$tmp.dir = '/tmp' # save costs on Azure
}

config$select.validset = function(config) {
  cat(date(), 'Selecting validset\n')

  if (config$validation.scheme == 'none') {
    return (0)
  }

  # NOTE: code duplication with preprocess.users ...
  colClasses = c('character', 'Date', 'character', 'Date', 'factor', 'numeric', 'factor', 'numeric', 'factor', 'factor', 'factor', 'factor', 'factor', 'factor', 'factor', 'factor')
  dtr = fread(paste0(config$datadir, '/train_users_2.csv'), colClasses = colClasses)
  dtr[, timestamp_first_active.d := as.Date(timestamp_first_active, format = '%Y%m%d')]
  dtr[date_first_booking == '', date_first_booking := NA]
  dtr[, date_first_booking := as.Date(date_first_booking)]
  dtr[, days.to.book := as.numeric(date_first_booking - timestamp_first_active.d)]
  if (config$fix.all) {
    dtr[days.to.book > config$max.days.to.book, country_destination := 'NDF']
  }
  dtr = dtr[order(timestamp_first_active.d, decreasing = T)]

  va.size = round(config$validation.frac * nrow(dtr))
  
  if (config$validation.scheme == 'random') {
    va.idx = sample(nrow(dtr), va.size)
  } else if (config$validation.scheme == 'last') {
    va.idx = 1:va.size
    cat('NOTE: last validation date is', as.character(dtr$timestamp_first_active.d[va.size]), '(may be partial)\n')
  } else {
    stop('Unexpected validation.scheme')
  }

  # Stratifying while doing things like 'last' introduces false signal, because we oversample NDF 
  # for example, and this looks as if recent dates had fewer NDF cases. So we must remove the whole
  # period from the training set, and then we can select a subset or more generally resample the
  # held-out indexes.
  
  tr.idx = (1:nrow(dtr))[-va.idx]

  if (config$stratify.validset) {
    nr.wanted.total = length(va.idx)
    for (i in 1:length(config$label.coding)) {
      nr.wanted.total = min(nr.wanted.total, sum(dtr$country_destination[va.idx] == config$label.coding[i]) / config$yp.te[i])
    }

    cand.idx = va.idx
    va.idx = NULL
    
    for (i in 1:length(config$label.coding)) {
      cand.idx.i = cand.idx[dtr$country_destination[cand.idx] == config$label.coding[i]]
      nr.wanted.i = round(config$yp.te[i] * nr.wanted.total)
      nr.to.sample.i = min(nr.wanted.i, length(cand.idx.i)) # FIXME warn?
      va.idx = c(va.idx, sample(cand.idx.i, nr.to.sample.i))
    }
  }
  
  tr.users = dtr$id[tr.idx]
  va.users = dtr$id[va.idx]
  
  save(tr.users, va.users, file = paste0(config$tmp.dir, '/trainset-split.RData'))
  
  cat('NOTE: actual validation fraction selected', length(unique(va.users)) / nrow(dtr), 'with class dist:\n')
  xx = table(dtr$country_destination[va.idx])
  xx = round(sort(xx / sum(xx), decreasing = T) * 1000) / 1000
  print(xx)
  cat('\n')
}

config$preprocess.sessions = function(config) {
  cat(date(), 'Processing sessions\n')
  
  dse = fread(paste0(config$datadir, '/sessions.csv'))
  
  dse = dse[-which(user_id == '')] # ?!
  dse[action_type == 'message_post', action := 'message_post']
  dse[action_type == '', action_type := action]
  dse[action_detail == '', action_detail := action_type]
  dse[action == '-unknown-', action := '_unknown']
  dse[action_type == '-unknown-', action_type := '_unknown']
  dse[action_detail == '-unknown-', action_detail := '_unknown']
  dse[, act := paste(action, action_type, action_detail, sep = '.')]
  dse[, act2 := shift(act, 1, type = 'lag', fill = 'first'), by = user_id]
  dse[, act.pair := paste(act2, act, sep = '...')]
  dse[is.na(secs_elapsed), secs_elapsed := 0]
  
  # It's possible that session activity is largly dictated by device type
  dse[, device.group := 'unknown']
  dse[device_type %in% c('Mac Desktop', 'Windows Desktop', 'Linux Desktop'), device.group := 'Web']
  dse[device_type %in% c('iPhone', 'iPad Tablet'), device.group := 'iOS']
  dse[device_type %in% c('Android Phone', 'Android App Unknown Phone/Tablet'), device.group := 'Android']
  dse[, device.group := as.factor(device.group)]

  if (0 && (config$fix.all || config$fix.va.and.te)) {
    # If we fix the horizon, it makes sense to remove actions recorded after the max time to book.
    # But only if we have complete sessions so that the sec_elapsed field truly reflects calendar
    # time.... which doesn't seem to be the case. Anyway, worth a try.
    # NOTE: I tried, and it doesn't work. Probably since we can't recover session timestampts at all.
    
    dse[, cumdays := cumsum(secs_elapsed) / (24 * 3600), by = 'user_id']
  
    if (config$fix.all) {
      cat('NOTE: fixing all sessions as per fixed labels\n')
      dse = dse[cumdays < config$max.days.to.book]
    } else if (config$fix.va.and.te) {
      cat('NOTE: fixing valid and test set sessions as per hypothesis\n')
    
      load(file = paste0(config$tmp.dir, '/trainset-split.RData')) # => tr.users, va.users
      load(file = paste0(config$tmp.dir, '/pp-users.RData')) # => dat
      ids.to.fix = dat[!(id %in% tr.users), id]

      dse = dse[!(user_id %in% ids.to.fix) | (cumdays < config$max.days.to.book)]
      # This creates a wierd distortion which may or may not be a problem...
    }

    dse[, cumdays := NULL]
  }
  
  ssn.base = dse[, list(
    ssn.total.actions = .N, 
    ssn.unique.actions = length(unique(action)) + length(unique(action_type)) + length(unique(action_detail)),
    ssn.nr.devices = length(unique(device_type)),
    ssn.total.time = sum(secs_elapsed, na.rm = T), 
    ssn.time.q0   = min     (secs_elapsed,       na.rm = T),
    ssn.time.q25  = quantile(secs_elapsed, 0.25, na.rm = T),
    ssn.time.q50  = median  (secs_elapsed,       na.rm = T),
    ssn.time.q75  = quantile(secs_elapsed, 0.75, na.rm = T),
    ssn.time.q100 = max     (secs_elapsed,       na.rm = T)
  ), by = .(user_id)]

  save(ssn.base, file = paste0(config$tmp.dir, '/pp-ssn-base.RData'))
  
  # TODO: Umm... filter out in the below almost-degenerate features to speed up the 
  # training / reduce var later?
  
  if (1) {
    # counts, action
    ssn.ca = dcast(dse[, .N, by = .(user_id, action)], user_id ~ action)
    ssn.ca[is.na(ssn.ca)] = 0
    names(ssn.ca) = paste0('ssn.nra1.', names(ssn.ca))
    names(ssn.ca)[1] = 'user_id'
    save(ssn.ca, file = paste0(config$tmp.dir, '/pp-ssn-ca.RData'))
    
    # counts, action X action_type
    ssn.caa = dcast(dse[, .N, by = .(user_id, action, action_type)], user_id ~ action + action_type)
    ssn.caa[is.na(ssn.caa)] = 0
    names(ssn.caa) = paste0('ssn.caa.', names(ssn.caa))
    names(ssn.caa)[1] = 'user_id'
    save(ssn.caa, file = paste0(config$tmp.dir, '/pp-ssn-caa.RData'))
    
    # counts, action X action_type X action_detail X device.group
    ssn.caaad = dcast(dse[, .N, by = .(user_id, action, action_type, action_detail, device.group)], user_id ~ action + action_type + action_detail + device.group)
    ssn.caaad[is.na(ssn.caaad)] = 0
    names(ssn.caaad) = paste0('ssn.caaad.', names(ssn.caaad))
    names(ssn.caaad)[1] = 'user_id'
    save(ssn.caaad, file = paste0(config$tmp.dir, '/pp-ssn-caaad.RData'))
  }
  
  # counts, action X action_type X action_detail
  ssn.caaa = dcast(dse[, .N, by = .(user_id, act)], user_id ~ act)
  ssn.caaa[is.na(ssn.caaa)] = 0
  names(ssn.caaa) = paste0('ssn.caaa.', names(ssn.caaa))
  names(ssn.caaa)[1] = 'user_id'
  if (0) {
    load('/tmp/pp-users.RData')
    dat = dat[is.test == F & dac.year == 14]
    dat = merge(dat, ssn.base, by.x = 'id', by.y = 'user_id', all.x = F)
    dat = merge(dat, ssn.caaa, by.x = 'id', by.y = 'user_id', all.x = F)
    pfi = grep('^ssn.caaa.', names(dat))
    pf = names(dat)[pfi]
    y = as.numeric(dat$country_destination != 'NDF')
    pv = rep(NA, length(pf))
    for (i in 1:length(pfi)) {
      x = unlist(dat[, pfi[i], with = F])
      if (length(unique(x)) > 1) {
        pv[i] = chisq.test(table(x > 0, y))$p.value
      }
    }
    pv[is.na(pv)] = 0
    ssn.act.pvs = data.frame(pf, pv)
    save(ssn.act.pvs, file = paste0(config$tmp.dir, '/ssn-act-marginal-pvs.RData'))
  } else if (1) {
    load(file = paste0(config$tmp.dir, '/ssn-act-marginal-pvs.RData')) # => ssn.act.pvs
    thresh = 1e-8
    cat('NOTE: removing', mean(ssn.act.pvs$pv > thresh), 'of the ssn caaa features\n')
    ssn.caaa = ssn.caaa[, -which(names(ssn.caaa) %in% ssn.act.pvs$pf[ssn.act.pvs$pv > thresh]), with = F]
  }
  save(ssn.caaa, file = paste0(config$tmp.dir, '/pp-ssn-caaa.RData'))
  
  # fractional frequencies, action X action_type X action_detail
  ssn.faaa = merge(ssn.caaa, ssn.base[, .(user_id, ssn.total.actions)], by = 'user_id')
  ssn.faaa.user_id = ssn.faaa$user_id
  ssn.faaa[, user_id := NULL]
  ssn.faaa = ssn.faaa / ssn.faaa$ssn.total.actions
  ssn.faaa[, ssn.total.actions := NULL]
  ssn.faaa[, user_id := ssn.faaa.user_id]
  names(ssn.faaa) = paste0('ssn.faaa.', substr(names(ssn.faaa), 10, 4000))
  names(ssn.faaa)[ncol(ssn.faaa)] = 'user_id'
  save(ssn.faaa, file = paste0(config$tmp.dir, '/pp-ssn-faaa.RData'))

  if (0) {
    # Add PCs of the (perhaps too high dimensional for modeling) session features
    prcomp.res = prcomp(ssn.caaa[, 2:ncol(ssn.caaa), with = F], center = T, scale. = T)
    
    if (0) {
      plot(prcomp.res, type = 'l', log = 'y', npcs = 150) # => take 50, or 20
      plot(prcomp.res$x[, 1], prcomp.res$x[, 2], pch = '.', log = 'xy')
      load('/tmp/pp-users.RData')
      dat = merge(dat, data.frame(user_id = ssn.caaa$user_id, prcomp.res$x), by.x = 'id', by.y = 'user_id', all.x = F)
      plot(dat$PC3, dat$PC4, col = 1 + (dat$country_destination == 'NDF'), pch = '.', xlim = c(-20, 20), ylim = c(-1, 50))
    }
    
    ssn.caaa.pcs = data.table(user_id = ssn.faaa$user_id, prcomp.res$x[, 1:50])
    names(ssn.caaa.pcs)[-1] = paste0('ssn.caaa.', names(ssn.caaa.pcs)[-1])
    save(ssn.caaa.pcs, file = paste0(config$tmp.dir, '/pp-ssn-caaa-pcs.RData'))
  }
  
  if (0) {
    # total time, action X action_type X action_detail
    ssn.taaa = dcast(dse[, sum(secs_elapsed / 1e6, na.rm = T), by = .(user_id, act)], user_id ~ act)
    ssn.taaa[is.na(ssn.taaa)] = 0
    names(ssn.taaa) = paste0('ssn.taaa.', names(ssn.taaa))
    names(ssn.taaa)[1] = 'user_id'
    save(ssn.taaa, file = paste0(config$tmp.dir, '/pp-ssn-taaa.RData'))
    
    # fractional time, action X action_type X action_detail
    ssn.ftaaa = merge(ssn.taaa, ssn.base[, .(user_id, ssn.total.time)], by = 'user_id')
    ssn.ftaaa.user_id = ssn.ftaaa$user_id
    ssn.ftaaa[, user_id := NULL]
    ssn.ftaaa = ssn.ftaaa / ssn.ftaaa$ssn.total.time
    ssn.ftaaa[, ssn.total.time := NULL]
    ssn.ftaaa[, user_id := ssn.ftaaa.user_id]
    names(ssn.ftaaa) = paste0('ssn.ftaaa.', substr(names(ssn.ftaaa), 10, 4000))
    names(ssn.ftaaa)[ncol(ssn.ftaaa)] = 'user_id'
    save(ssn.ftaaa, file = paste0(config$tmp.dir, '/pp-ssn-ftaaa.RData'))
  
    # "2-gram" actions (indicators of pairs of consecutive session actions)
    load(file = paste0(config$tmp.dir, '/pp-users.RData')) # => dat
    dse.act.pair = unique(dse[, .(user_id, act.pair)]) # NOTE: this will only allow us to check if a user did an action or not, not how many time they did it
    dse.act.pair = merge(dse.act.pair, dat[, .(id, booked = as.integer(country_destination != 'NDF'))], by.x = 'user_id', by.y = 'id')
    dse.act.pair = dse.act.pair[!is.na(booked)]
    
    # NOTE: there are too many of these, so we filter (using the entire training set, so might overfit a little)
    act.pair.0 = dse.act.pair[booked == 0, .N, by = .(act.pair)]
    act.pair.1 = dse.act.pair[booked == 1, .N, by = .(act.pair)]
    act.pair.tables = merge(act.pair.0, act.pair.1, by = 'act.pair', all = T)
    act.pair.tables[is.na(act.pair.tables)] = 0
    N0 = sum(dse.act.pair$booked == 0)
    N1 = sum(dse.act.pair$booked == 1)
    act.pair.tables[, p := chisq.test(matrix(c(N0 - N.x, N.x, N1 - N.y, N.y), nrow = 2))$p.value, by = 'act.pair']
    act.pair.tables[, q := p.adjust(p, method = 'BH')]
    act.pair.tables = act.pair.tables[(q < 0.1) & (N.x + N.y > 100)]
  
    ssn.cap = dcast(dse[act.pair %in% act.pair.tables$act.pair, .N, by = .(user_id, act.pair)], user_id ~ act.pair)
    ssn.cap[is.na(ssn.cap)] = 0
    names(ssn.cap) = paste0('ssn.cap.', names(ssn.cap))
    names(ssn.cap)[1] = 'user_id'
    save(ssn.cap, file = paste0(config$tmp.dir, '/pp-ssn-cap.RData'))
  }
}

config$preprocess.misc = function(config) {
  # Buckets

  # NOTE: it doesn't look like this reflects the testset distribution at all! e.g., here DE is more
  # popular than FR, IT, GB, ES, CA ?? I suppose the file is about the population of each country, 
  # not the population of airbnb users who booked in that country... it therefore seems pretty 
  # useless for the present task. Go figure...
  
  if (0) {
    cat(date(), 'Processing buckets\n')
    
    buckets = fread(paste0(config$datadir, '/age_gender_bkts.csv'))
    bage = data.frame(t(apply(t(as.matrix(buckets[ , strsplit(age_bucket, '[-+]')], nrow = 2)), 1, as.numeric)))
    names(bage) = c('age.lo', 'age.hi')
    bage$age.hi[bage$age.hi == 100] = 200
    buckets = cbind(buckets, bage)
    
    # It doesn't seem like there is a big difference between genders, and gender info for most users 
    # is missing anyway, so merge (sum over gender)
    buckets = dcast(buckets[, .(country_destination, age.lo, age.hi, population_in_thousands)], age.lo + age.hi ~ country_destination, sum)
    buckets[, bucket.id := 1:nrow(buckets)]
    
    # For each bucket, compute the top 5 destinations
    buckets = cbind(buckets[, .(bucket.id, age.lo, age.hi)], top.countries = t(apply(buckets[, .(US, FR, IT, GB, ES, CA, DE, NL, AU, PT)], 1, order, decreasing = T))[, 1:5])
  }
  
  # Countries
  # Also seems useless.
}

config$preprocess.users = function(config) {
  cat(date(), 'Processing users\n')
  
  # fread will ignore most of these types, but to fix some long ints that should be strings:
  colClasses = c('character', 'Date'              , 'character'           , 'Date'            , 'factor', 'numeric', 'factor'     , 'numeric'  , 'factor' , 'factor'        , 'factor'           , 'factor'               , 'factor'  , 'factor'         , 'factor'     , 'factor'           )
  #              id         , date_account_created, timestamp_first_active, date_first_booking, gender  , age      , signup_method, signup_flow, language , affiliate_channel, affiliate_provider, first_affiliate_tracked, signup_app, first_device_type, first_browser, country_destination
  
  dtr = fread(paste0(config$datadir, '/train_users_2.csv'), colClasses = colClasses      )
  dte = fread(paste0(config$datadir, '/test_users.csv'   ), colClasses = colClasses[1:15])

  # Merge train and test so that processing is identical
  dte[, country_destination := NA]
  dte[, is.test := T]
  dtr[, is.test := F]
  dat = rbind(dtr, dte)

  # Date stuff
  dat[, date_account_created := as.Date(date_account_created)]
  dat[, dac.year  := as.integer(format(date_account_created, '%y'))]
  dat[, dac.month := as.integer(format(date_account_created, '%m'))]
 #dat[, dac.day   := as.integer(format(date_account_created, '%d'))]
  dat[, dac.wday  := wday(date_account_created)]
  
  dat[, timestamp_first_active.d := as.Date(timestamp_first_active, format = '%Y%m%d')]
  dat[, timestamp_first_active.t := as.integer(substr(timestamp_first_active, 9, 10))]
  dat[, timestamp_first_active.t2 := 60 * timestamp_first_active.t + as.integer(substr(timestamp_first_active, 11, 12))]
  dat[, tfa.year  := as.integer(format(timestamp_first_active.d, '%y'))]
  dat[, tfa.month := as.integer(format(timestamp_first_active.d, '%m'))]
 #dat[, tfa.day   := as.integer(format(timestamp_first_active.d, '%d'))]
  dat[, tfa.wday  := wday(timestamp_first_active.d)]

  dat[, time.to.first.use := as.integer(timestamp_first_active.d - date_account_created)]
  
  dat[date_first_booking == '', date_first_booking := NA]
  dat[, date_first_booking := as.Date(date_first_booking)]
  dat[, days.to.book := as.numeric(date_first_booking - timestamp_first_active.d)]

  if (config$fix.all) {
    # Assuming they used a fixed horizon for the testset, which was just different than the one
    # used in the trainset, fix the horizon:
    dat[days.to.book > config$max.days.to.book, country_destination := 'NDF']

    cat('NOTE: fixing all labels to match the hypothesized testset label generation process\n')
    xx = table(dat$country_destination)
    xx = round(sort(xx / sum(xx), decreasing = T) * 1000) / 1000
    print(xx)
    cat('\n')
  } else if (config$fix.va.and.te) {
    load(file = paste0(config$tmp.dir, '/trainset-split.RData')) # => tr.users, va.users
  
    dat[(id %in% va.users) & (days.to.book > config$max.days.to.book), country_destination := 'NDF']
    
    cat('NOTE: fixing validset labels to match the hypothesized testset label generation process\n')
    xx = table(dat[id %in% va.users, country_destination])
    xx = round(sort(xx / sum(xx), decreasing = T) * 1000) / 1000
    print(xx)
    cat('\n')
  }

  # Clean age by removing fishy values
  # NOTE: this assumes XGB is the model
  dat[, age.missing := as.numeric(age < 15 | age > 95)]
  dat[, age.clean := age]
  dat[age < 15 | age > 95, age.clean := NA]
  #dat[, age.clean.cat := as.factor(floor((age.clean - 15) / 10))]
  
  #
  # Handle categorical variables
  #

  #cat.names = c('gender', 'language', 'signup_method', 'signup_app', 'signup_flow', 'first_device_type', 'first_browser', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked')
  #for (f in cat.names) {
  #  barplot(table(as.numeric(dtr$country_destination != 'NDF'), dtr[[f]]), main = f)
  #}

  cat.names = names(dat)[lapply(dat, class) == 'character']
  for (f in cat.names) {
    idx = which(dat[, f] == '-unknown-')
    if (length(idx) > 0) {
      dat[idx, f] = NA
    }
  }
  
  sms.encode = function (x, y) {
    train.idx = !is.na(y)
    y.train = as.numeric(y[train.idx])
    x.train = x[train.idx]
    x.test = x[!train.idx]
    cv.folds = createFolds(as.factor(y.train), k = 10)
    
    x.train.enc = rep(NA, length(y.train))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      bad.idx = which(!(x.train[idx] %in% unique(x.train[-idx])))
      if (length(bad.idx) > 0) {
        idx2 = idx[-bad.idx]
      } else {
        idx2 = idx
      }
      x.train.enc[idx2] = predict(lm(y ~ x - 1, data.frame(y = y.train[-idx], x = x.train[-idx])), data.frame(x = x.train[idx2]))
    }
    
    bad.idx = which(!(x.test %in% unique(x.train)))
    if (length(bad.idx) > 0) {
      x.test.enc = rep(NA, length(x.test))
      x.test.enc[-bad.idx] = predict(lm(y ~ x - 1, data.frame(y = y.train, x = x.train)), data.frame(x = x.test[-bad.idx]))
    } else {
      x.test.enc = predict(lm(y ~ x - 1, data.frame(y = y.train, x = x.train)), data.frame(x = x.test))
    }
    
    x.enc = rep(NA, length(y))
    x.enc[ train.idx] = x.train.enc
    x.enc[!train.idx] = x.test.enc
    return (x.enc)
  }

  dat[, country_destination     := factor(country_destination, levels = config$label.coding)]
  dat[, gender                  := as.factor(gender                 )]
  dat[, language                := as.factor(language               )]
  dat[, signup_method           := as.factor(signup_method          )]
  dat[, signup_flow             := as.factor(signup_flow            )]
  dat[, signup_app              := as.factor(signup_app             )]
  dat[, affiliate_provider      := as.factor(affiliate_provider     )]
  dat[, affiliate_channel       := as.factor(affiliate_channel      )]
  dat[, first_affiliate_tracked := as.factor(first_affiliate_tracked)]
  dat[, first_device_type       := as.factor(first_device_type      )]
  dat[, first_browser           := as.factor(first_browser          )]
  dat[, dac.month.cat           := as.factor(dac.month              )]
  dat[, tfa.year.cat            := as.factor(tfa.year               )]
  dat[, tfa.month.cat           := as.factor(tfa.month              )]
  dat[, tfa.wday.cat            := as.factor(tfa.wday               )]
  dat[, tfa.t.cat               := as.factor(timestamp_first_active.t)]

  age.new = dat$age
  age.new[age.new >= 2000] = 2000
  age.new[age.new >= 1900 & age.new < 2000] = 1900
  age.new[age.new >= 110 & age.new < 1900] = 110
  age.new[age.new >= 105 & age.new < 110] = 105
  age.new[age.new >= 90 & age.new < 105] = 100
  age.new[age.new >= 75 & age.new < 90] = 80
  age.new[age.new >= 61 & age.new < 75] = 65
  age.new[age.new >= 51 & age.new < 61] = 55
  age.new[age.new >= 41 & age.new < 51] = 45
  age.new[age.new >= 33 & age.new < 41] = 35
  age.new[age.new >= 28 & age.new < 33] = 30
  age.new[age.new >= 24 & age.new < 28] = 25
  age.new[age.new >= 22 & age.new < 24] = 22
  age.new[age.new >= 19 & age.new < 22] = 20
  age.new[age.new >= 15 & age.new < 19] = 18
  age.new[age.new < 15] = 0
  dat[, age.cat := as.factor(age.new)]
  
  # Language pertains to countries in "other" (includes South American es and pt)
  dat[, language.other := as.numeric(language %in% c('zh', 'es', 'ko', 'ru', 'ja', 'pt', 'sv', 'tr', 'da', 'pl', 'no', 'cs', 'el', 'th', 'hu', 'id', 'fi', 'is', 'hr'))]

  # Some important interactions
  dat[, ageXsignup_method := interaction(age.cat, signup_method)]
  dat[, ageXsignup_flow := interaction(age.cat, signup_flow)]
  dat[, signup_methodXsignup_flow := interaction(signup_method, signup_flow)]
  
  # The faction of users with each response level in each feature level
  sms.encode.all = function(ylevel) {
    dat[, paste0('age.sms.'                     , ylevel) := sms.encode(age.cat                , country_destination == ylevel)]
    dat[, paste0('signup_method.sms.'           , ylevel) := sms.encode(signup_method          , country_destination == ylevel)]
    dat[, paste0('signup_flow.sms.'             , ylevel) := sms.encode(signup_flow            , country_destination == ylevel)]
    dat[, paste0('language.sms.'                , ylevel) := sms.encode(language               , country_destination == ylevel)]
    dat[, paste0('affiliate_provider.sms.'      , ylevel) := sms.encode(affiliate_provider     , country_destination == ylevel)]
    dat[, paste0('affiliate_channel.sms.'       , ylevel) := sms.encode(affiliate_channel      , country_destination == ylevel)]
    dat[, paste0('first_affiliate_tracked.sms.' , ylevel) := sms.encode(first_affiliate_tracked, country_destination == ylevel)]
    dat[, paste0('first_device_type.sms.'       , ylevel) := sms.encode(first_device_type      , country_destination == ylevel)]
    dat[, paste0('first_browser.sms.'           , ylevel) := sms.encode(first_browser          , country_destination == ylevel)]

    dat[, paste0('ageXsignup_method.sms.'         , ylevel) := sms.encode(ageXsignup_method        , country_destination == ylevel)]
    dat[, paste0('ageXsignup_flow.sms.'           , ylevel) := sms.encode(ageXsignup_flow          , country_destination == ylevel)]
    dat[, paste0('signup_methodXsignup_flow.sms.' , ylevel) := sms.encode(signup_methodXsignup_flow, country_destination == ylevel)]

    if (ylevel != 'NDF') {
      dat[, paste0('tfa.month.sms.' , ylevel) := sms.encode(tfa.month.cat   , country_destination == ylevel)]
      dat[, paste0('tfa.t.sms.'     , ylevel) := sms.encode(tfa.t.cat       , country_destination == ylevel)]
    }
  }

  # NOTE: I didn't want to do this, because it means I'll need separate data per stage, alas...
  if (config$main.approach %in% c('allclass', 'top5', 'NDF/booked', 'NDF/US/rest')) {
    sms.encode.all('NDF')
  }
  if (config$main.approach %in% c('allclass', 'top5', 'NDF/US/rest', 'US/notUS|booked', 'destination|booked')) {
    sms.encode.all('US')
  }
  if (config$main.approach %in% c('allclass', 'top5', 'destination|booked', 'destination|bookedNotUS', 'other/FR/IT|oneOfThese')) {
    sms.encode.all('other')
  }
  if (config$main.approach %in% c('allclass', 'top5', 'destination|booked', 'destination|bookedNotUS', 'other/FR/IT|oneOfThese')) {
    sms.encode.all('FR')
  }
  if (config$main.approach %in% c('allclass', 'destination|booked', 'destination|bookedNotUS')) {
    sms.encode.all('IT')
  }
  if (config$main.approach %in% c('allclass', 'destination|booked', 'destination|bookedNotUS')) {
    sms.encode.all('GB')
    sms.encode.all('ES')
    sms.encode.all('CA')
    sms.encode.all('DE')
  }

  if (0) {
    # These encode the destination-rareness of each of the categorical feature's levels
    dat[, age.smsr                     := sms.encode(age.cat                , country_destination)]
    dat[, language.smsr                := sms.encode(language               , country_destination)]
    dat[, signup_flow.smsr             := sms.encode(signup_flow            , country_destination)]
    dat[, affiliate_channel.smsr       := sms.encode(affiliate_channel      , country_destination)]
    dat[, affiliate_provider.smsr      := sms.encode(affiliate_provider     , country_destination)]
    dat[, first_affiliate_tracked.smsr := sms.encode(first_affiliate_tracked, country_destination)]
    dat[, first_device_type.smsr       := sms.encode(first_device_type      , country_destination)]
    dat[, first_browser.smsr           := sms.encode(first_browser          , country_destination)]
    dat[, dac.month.smsr               := sms.encode(dac.month.cat          , country_destination)]
  }

 #dat[, age.cat                 := NULL]
 #dat[, gender                  := NULL]
 #dat[, language                := NULL]
 #dat[, signup_method           := NULL]
  dat[, signup_flow             := NULL]
  dat[, signup_app              := NULL]
  dat[, affiliate_provider      := NULL]
  dat[, affiliate_channel       := NULL]
  dat[, first_affiliate_tracked := NULL]
  dat[, first_device_type       := NULL]
  dat[, first_browser           := NULL]
  dat[, dac.month.cat           := NULL]
  dat[, tfa.year.cat            := NULL]
  dat[, tfa.month.cat           := NULL]
  dat[, tfa.wday.cat            := NULL]
  dat[, tfa.t.cat               := NULL]
  
  if (config$main.approach == 'NDF/booked') {
    dat[, timestamp_first_active.t := NULL]
    dat[, timestamp_first_active.t2 := NULL]
  }
  
  save(dat, file = paste0(config$tmp.dir, '/pp-users.RData'))
}

config$compile.data = function(config) {
  cat(date(), 'Compiling data sets\n')

  # Load users data
  load(file = paste0(config$tmp.dir, '/pp-users.RData')) # => dat
  
  # Load and merge sessions data
  cat(date(), 'Merging session data\n')
    
  if ('base' %in% config$session.features) {
    load(file = paste0(config$tmp.dir, '/pp-ssn-base.RData')) # => ssn.base
    dat = merge(dat, ssn.base, by.x = 'id', by.y = 'user_id', all.x = T)
    dat[, have.ssn := as.numeric(!is.na(ssn.total.actions) & ssn.total.actions > 0)]
  }
  
  if ('ca' %in% config$session.features) {
    load(file = paste0(config$tmp.dir, '/pp-ssn-ca.RData')) # => ssn.ca
    dat = merge(dat, ssn.ca, by.x = 'id', by.y = 'user_id', all.x = T)
  }
  
  if ('caa' %in% config$session.features) {
    load(file = paste0(config$tmp.dir, '/pp-ssn-caa.RData')) # => ssn.caa
    dat = merge(dat, ssn.caa, by.x = 'id', by.y = 'user_id', all.x = T)
  }
  
  if ('caaa' %in% config$session.features) {
    load(file = paste0(config$tmp.dir, '/pp-ssn-caaa.RData')) # => ssn.caaa
    dat = merge(dat, ssn.caaa, by.x = 'id', by.y = 'user_id', all.x = T)
    
    dat[, ssn.hostness := 
      #ssn.caaa.manage_listing.view.manage_listing +
      #ssn.caaa.index.view.your_listings +
      ssn.caaa.ajax_check_dates.click.change_contact_host_dates +
      ssn.caaa.create.submit.create_listing +
      ssn.caaa.manage_listing.view.manage_listing +
      ssn.caaa.index.data.reservations]
    
    dat[, ssn.seriousness := 
      (ssn.caaa.search_results.click.view_search_results                > 0) +
      (ssn.caaa.index.view.view_search_results                          > 0) +
      (ssn.caaa.search.click.view_search_results                        > 0) +
      (ssn.caaa.show.show.show                                          > 0) +
      (ssn.caaa.ajax_refresh_subtotal.click.change_trip_characteristics > 0) +
      (ssn.caaa.similar_listings.data.similar_listings                  > 0) +
      (ssn.caaa.reviews.data.listing_reviews                            > 0)]
    
    dat[, ssn.activeness := pmin(
      ssn.caaa.this_hosting_reviews.click.listing_reviews_page + 
      ssn.caaa.cancellation_policies.view.cancellation_policies +
      ssn.caaa.message_post.message_post.message_post +
      ssn.caaa.message_to_host_focus.click.message_to_host_focus + 
      ssn.caaa.message_to_host_change.click.message_to_host_change, 3)]
    
    dat[, ssn.bookness := 
      (ssn.caaa.at_checkpoint.booking_request.at_checkpoint > 0) +
     #(ssn.caaa.click.click.complete_booking                > 0) +
      (ssn.caaa.agree_terms_check._unknown._unknown         > 0) +
      (ssn.caaa.requested.submit.post_checkout_action       > 0) +
      (ssn.caaa.requested.view.p5                           > 0) +
      (ssn.caaa.apply_reservation.submit.apply_coupon       > 0) +
      (ssn.caaa.pay._unknown._unknown                       > 0) +
      (ssn.caaa.receipt.view.guest_receipt                  > 0) +
      (ssn.caaa.email_itinerary_colorbox._unknown._unknown  > 0) +
      (ssn.caaa.travel_plans_current.view.your_trips        > 0) +
      (ssn.caaa.pending.booking_request.pending             > 0) +
      (ssn.caaa.add_guests._unknown._unknown                > 0) +
      (ssn.caaa.impressions.view.p4                         > 0) +
      (ssn.caaa.reviews_new._unknown._unknown               > 0)]
  }

  if ('baaa' %in% config$session.features) {
    load(file = paste0(config$tmp.dir, '/pp-ssn-caaa.RData')) # => ssn.caaa
    uid = ssn.caaa[, user_id]
    ssn.caaa = as.data.table((ssn.caaa[, 2:ncol(ssn.caaa), with = F] > 0) + 0)
    ssn.caaa[, user_id := uid]
    dat = merge(dat, ssn.caaa, by.x = 'id', by.y = 'user_id', all.x = T)
  }
  
  if ('caaad' %in% config$session.features) {
    load(file = paste0(config$tmp.dir, '/pp-ssn-caaad.RData')) # => ssn.caaad
    dat = merge(dat, ssn.caaad, by.x = 'id', by.y = 'user_id', all.x = T)
  }
  
  if ('taaa' %in% config$session.features) {
    load(file = paste0(config$tmp.dir, '/pp-ssn-taaa.RData')) # => ssn.taaa
    dat = merge(dat, ssn.taaa, by.x = 'id', by.y = 'user_id', all.x = T)
  }
  
  if ('faaa' %in% config$session.features) {
    load(file = paste0(config$tmp.dir, '/pp-ssn-faaa.RData')) # => ssn.faaa
    dat = merge(dat, ssn.faaa, by.x = 'id', by.y = 'user_id', all.x = T)
  }    

  if ('ftaaa' %in% config$session.features) {
    load(file = paste0(config$tmp.dir, '/pp-ssn-ftaaa.RData')) # => ssn.ftaaa
    dat = merge(dat, ssn.ftaaa, by.x = 'id', by.y = 'user_id', all.x = T)
  }
  
  if ('caaa.pcs' %in% config$session.features) {
    load(file = paste0(config$tmp.dir, '/pp-ssn-caaa-pcs.RData')) # => ssn.caaa.pcs
    dat = merge(dat, ssn.caaa.pcs, by.x = 'id', by.y = 'user_id', all.x = T)
  }

  if ('cap' %in% config$session.features) {
    load(file = paste0(config$tmp.dir, '/pp-ssn-cap.RData')) # => ssn.cap
    dat = merge(dat, ssn.cap, by.x = 'id', by.y = 'user_id', all.x = T)
  }
  
  # Important: reordering the data in a standard fixed order (essential to easily combine and
  # compare predictions on the validation and test sets for example)
  dat = dat[order(id)]
  # (now don't touch dat anymore...)
  
  # Extract data for modeling
  cat(date(), 'Extracting data for modeling\n')
  
  if (config$simplify.model) {
    cat('NOTE: simplifying users features\n')
    modeling.forumla = ~ age.clean + language + tfa.year + signup_app + signup_method + 
      signup_flow.int + first_device_type.int + first_browser.int - 1
  } else {
    modeling.forumla = ~ . - 1 - id - date_account_created - timestamp_first_active - 
      timestamp_first_active.d - country_destination - is.test - days.to.book - date_first_booking
  }
  
  modeling.dat = predict(dummyVars(modeling.forumla, data = dat), newdata = dat)
  feature.names = colnames(modeling.dat)
  
  if (config$experiment.dupfeat) {
    nr.dups = round(1 / config$xgb.colsample_bytree) - 1
    if (nr.dups > 0) {
      cat('EXPERIMENT: duplicating important features', nr.dups, 'times\n')
      
      modeling.forumla2 = ~ -1 + gender + age.clean + age.missing + language.int + 
        signup_method + signup_flow.int + affiliate_channel.int + affiliate_provider.int +
        first_affiliate_tracked.int + first_device_type.int + 
        timestamp_first_active.t + 
        time.to.first.use + first_browser.int +
        ssn.total.actions + ssn.unique.actions + ssn.nr.devices + 
        ssn.total.time + ssn.time.q0 + ssn.time.q25 + ssn.time.q50 + ssn.time.q75 + ssn.time.q100
      
      dups = predict(dummyVars(modeling.forumla2, data = dat), newdata = dat)
      dups.names = names(dups)
      
      for (i in 1:nr.dups) {
        prfx = paste0('dup', i, '.')
        names(dups) = paste0(prfx, dups.names)
        modeling.dat = cbind(modeling.dat, dups)
      }
      feature.names = colnames(modeling.dat)
    }
  }
  
  # Recode missing values for xgb
  modeling.dat[is.na(modeling.dat)] = config$recode.na.as
  
  # Transform train and test sets to XGB format and save
  cat(date(), 'Saving datasets\n')
  tr.idx = which(!dat$is.test)
  te.idx = which( dat$is.test)
  xgb.trainset = xgb.DMatrix(modeling.dat[tr.idx, ], missing = config$recode.na.as) # NOTE: will add labels and weights later on
  xgb.testset  = xgb.DMatrix(modeling.dat[te.idx, ], missing = config$recode.na.as)
  xgb.DMatrix.save(xgb.trainset, paste0(config$tmp.dir, '/xgb-trainset-', config$data.tag, '.data'))
  xgb.DMatrix.save(xgb.testset , paste0(config$tmp.dir, '/xgb-testset-' , config$data.tag, '.data'))
  
  # Finally, store some ancillary data we will need later
  ancillary = list()
  ancillary$pp.config = config
  ancillary$feature.names = feature.names
  ancillary$train = dat[tr.idx, .(id, country_destination, timestamp_first_active.d, days.to.book, language)]
  ancillary$test = dat[te.idx, .(id, language)]
  save(ancillary, file = paste0(config$tmp.dir, '/pp-data-ancillary-', config$data.tag, '.RData'))
}

config$finalize.data = function(config) {
  cat(date(), 'Finalizing data\n')
  load(paste0(config$tmp.dir, '/pp-data-ancillary-', config$data.tag, '.RData')) # => ancillary
  
  # Recode the target variable as [0, nr_classes] for xgb
  recodes = paste0('\'', config$label.coding, '\'=', (1:length(config$label.coding)) - 1, collapse = '; ')
  ancillary$train[, label := as.numeric(as.character(recode(country_destination, recodes)))]
  ancillary$train[, label.orig := label]
  if (config$main.approach == 'NDF/booked') {
    recodes = paste0('\'', config$label.coding, '\'=', c(0, rep(1, 11)), collapse = '; ')
    ancillary$train[, label := as.numeric(as.character(recode(country_destination, recodes)))]
  } else if (config$main.approach == 'NDF/US/rest') {
    recodes = paste0('\'', config$label.coding, '\'=', c(0, 1, rep(2, 11)), collapse = '; ')
    ancillary$train[, label := as.numeric(as.character(recode(country_destination, recodes)))]
  } else if (config$main.approach == 'top5') {
    recodes = paste0('\'', config$label.coding, '\'=', c(0:4, rep(-1, 7)), collapse = '; ')
    ancillary$train[, label := as.numeric(as.character(recode(country_destination, recodes)))]
  } else if (config$main.approach == 'US/notUS|booked') {
    recodes = paste0('\'', config$label.coding, '\'=', c(-1, 0, rep(1, 11)), collapse = '; ')
    ancillary$train[, label := as.numeric(as.character(recode(country_destination, recodes)))]
    # Instead of dropping the -1's, try to keep them in and draw marginally uninformative labels for them
    #yp.tr = config$yp.tr[-1] / sum(config$yp.tr[-1])
    #yp.tr = c(yp.tr[1], 1 - yp.tr[1])
    #ancillary$train[label == -1, label := as.numeric(sample(0:1, sum(ancillary$train[, label] == -1), replace = T, prob = yp.tr))]
  } else if (config$main.approach == 'destination|booked') {
    recodes = paste0('\'', config$label.coding, '\'=', c(-1, 0:10), collapse = '; ')
    ancillary$train[, label := as.numeric(as.character(recode(country_destination, recodes)))]
    # Instead of dropping the -1's, try to keep them in and draw marginally uninformative labels for them
    #yp.tr = config$yp.tr[-1] / sum(config$yp.tr[-1])
    #ancillary$train[label == -1, label := sample(0:11, sum(ancillary$train[, label] == -1), replace = T, prob = yp.tr)]
  } else if (config$main.approach == 'destination|bookedNotUS') {
    recodes = paste0('\'', config$label.coding, '\'=', c(-1, -1, 0:9), collapse = '; ')
    ancillary$train[, label := as.numeric(as.character(recode(country_destination, recodes)))]
  } else if (config$main.approach == 'other/FR/IT|oneOfThese') {
    recodes = paste0('\'', config$label.coding, '\'=', c(-1, -1, 0:2, rep(-1, 7)), collapse = '; ')
    ancillary$train[, label := as.numeric(as.character(recode(country_destination, recodes)))]
  } else {
    stop('Unexpected main.approach')
  }

  #
  # Mark examples we don't wish to train on
  #
  
  ancillary$train[, train.on := (label >= 0)]
  
  if (config$only.use.2014.data) {
    cat('NOTE: training only on samples from 2014\n')
    ancillary$train[timestamp_first_active.d < '2014-01-01', train.on := F]
  }
  
  #
  # Split to (actual) train and validation sets
  #

  if (config$do.manual.cv1) {
    start.valid.date = paste0('2014-', config$manual.cv.fold + 1, '-01')
    end.valid.date   = paste0('2014-', config$manual.cv.fold + 2, '-01')
    ancillary$train[timestamp_first_active.d >= start.valid.date & timestamp_first_active.d < end.valid.date, train.on := F]
    ancillary$train[timestamp_first_active.d >= start.valid.date & timestamp_first_active.d < end.valid.date, valid.on := T]
  } else if (config$validation.scheme != 'none') {
    load(file = paste0(config$tmp.dir, '/trainset-split.RData')) # => tr.users, va.users
    ancillary$train[id %in% va.users, train.on := F]
    ancillary$train[id %in% va.users, valid.on := T]
  }

  #
  # Assign training weights
  #
  
  ancillary$train[, weight := 1]
  
  yp.tr = table(ancillary$train[train.on == T, label.orig])
  yp.tr = yp.tr / sum(yp.tr)
  
  if (config$weighting.scheme == 'book') {
    cat('NOTE: Reweighting examples to adjust for train/test marginal booking probability\n')
    w.class = c(config$yp.te[1] / yp.tr[1], rep((1 - yp.te[1]) / (1 - yp.tr[1]), 11))
    ancillary$train[train.on == T, weight := w.class[label.orig + 1]]
  } else if (config$weighting.scheme == 'class') {
    cat('NOTE: Reweighting examples to adjust for train/test marginal class distr\n')
    w.class = config$yp.te / yp.tr
    ancillary$train[train.on == T, weight := w.class[label.orig + 1]]
  } else if (config$weighting.scheme == 'time') {
    cat('NOTE: Giving more recent examples more weight in training\n')
    stop('TODO')
  }

  # Finalize the xgb data
  trainset = xgb.DMatrix(paste0(config$tmp.dir, '/xgb-trainset-', config$data.tag, '.data'))
  setinfo(trainset, 'label', ancillary$train$label)
  if (config$weighting.scheme != 'none') {
    setinfo(trainset, 'weight', ancillary$train$weight)
  }
  
  # Hold out the validation set  
  if (config$do.manual.cv1 || config$validation.scheme != 'none') {
    validset = slice(trainset, which(ancillary$train$valid.on))
  } else {
    validset = NULL
  }
  
  # Bootstrap the trainset
  if (config$mode == 'bagging') {
    bs.idx = sample(which(ancillary$train$train.on), replace = T)
    trainset = slice(trainset, bs.idx)
    stop('TODO') # revive this if necessary: need to modify ancilllay accordingly
  } else {
    trainset = slice(trainset, which(ancillary$train$train.on))
  }

  return (list(trainset = trainset, validset = validset, ancillary = ancillary))
}

config$preprocess = function(config) {
  if (config$do.select.validset) {
    config$select.validset(config)
  } else {
    cat(date(), 'NOTE: using a previously selected validation set\n')
  }
  
  if (config$do.preprocess.users) {
    config$preprocess.users(config)
  } else {
    cat(date(), 'NOTE: using previously preprocessed users data\n')
  }
  
  if (config$do.preprocess.sessions) {
    config$preprocess.sessions(config)
  } else {
    cat(date(), 'NOTE: using previously preprocessed sessions data\n')
  }

  if (config$do.compile.data) {
    config$compile.data(config)
  } else {
    cat(date(), 'NOTE: using previously finalized data\n')
  }
}

# Expand a matrix of predicted class probabilities to (nr.users x 5) country names matrix
config$expand.pred = function(config, preds, langs, s1.preds = NULL) {
  if (config$naive.expand1) {
    ###### amazing that this is still often better...
    us.first = preds > 1 - config$two.class.thresh
    preds = cbind(
      ifelse(us.first, 'US', 'NDF'), 
      ifelse(us.first, 'NDF', 'US'), 
      'other', 'FR', 'IT'
    )
    
    return(preds)
    #####
  }
  
  if (config$naive.expand2) {
    # A very naive, hand crafted language based model, that I can't seem to train methodically
    first  = ifelse(preds > 1 - config$two.class.thresh, 'US', 'NDF')
    second = ifelse(preds > 1 - config$two.class.thresh, 'NDF', 'US')
    third = rep('other', length(first))
    third[langs == 'fr'] = 'FR'
    third[langs == 'de'] = 'DE'
    third[langs == 'it'] = 'IT'
    third[langs == 'no'] = 'GB'
    fourth = rep('FR', length(first))
    fourth[langs == 'fr'] = 'other'
    fourth[langs == 'es'] = 'ES'
    fourth[langs == 'de'] = 'other'
    fourth[langs == 'it'] = 'other'
    fourth[langs == 'ru'] = 'IT'
    fourth[langs == 'no'] = 'other'
    fifth = rep('IT', length(first))
    fifth[langs == 'de'] = 'ES'
    fifth[langs == 'it'] = 'ES'
    fifth[langs == 'es'] = 'FR'
    fifth[langs == 'ru'] = 'ES'
    fifth[langs == 'no'] = 'FR'
    
    preds = cbind(first, second, third, fourth, fifth)

    return(preds)
  }

  if (config$naive.expand3) {
    # Two threshold version of the naive language based model
    first  = rep('NDF', length(langs))
    second = rep('US', length(langs))
    third = rep('other', length(langs))
    third[langs == 'fr'] = 'FR'
    third[langs == 'de'] = 'DE'
    third[langs == 'it'] = 'IT'
    third[langs == 'no'] = 'GB'
    fourth = rep('FR', length(langs))
    fourth[langs == 'fr'] = 'other'
    fourth[langs == 'es'] = 'ES'
    fourth[langs == 'de'] = 'other'
    fourth[langs == 'it'] = 'other'
    fourth[langs == 'ru'] = 'IT'
    fourth[langs == 'no'] = 'other'
    fifth = rep('IT', length(langs))
    fifth[langs == 'de'] = 'ES'
    fifth[langs == 'it'] = 'ES'
    fifth[langs == 'es'] = 'FR'
    fifth[langs == 'ru'] = 'ES'
    fifth[langs == 'no'] = 'FR'

    idx2 = preds > 0.51
    idx3 = preds > 0.89 #0.87 and 0.90 were worse on the pub lb so far
    idx4 = preds > 0.95
    
    preds         = cbind(first, second, third, fourth, fifth)
    preds[idx2, ] = cbind(second, first, third, fourth, fifth)[idx2, ]
    preds[idx3, ] = cbind(second, third, first, fourth, fifth)[idx3, ]
    preds[idx4, ] = cbind(second, third, fourth, fifth, first)[idx4, ]
    
    return(preds)
  }

  if (config$naive.expand4) {
    #                 NDF    US other    FR    IT    GB    ES    CA    DE    NL    AU    PT
    #     en        0.580 0.295 0.047 0.024 0.013 0.011 0.011 0.007 0.005 0.004 0.003 0.001
    #     zh        0.681 0.251 0.048 0.008 0.003 0.003 0.002 0.001 0.001 0.000 0.001 0.000
    #     fr        0.660 0.217 0.016 0.059 0.009 0.010 0.009 0.006 0.007 0.003 0.002 0.002
    #     es        0.699 0.181 0.051 0.020 0.010 0.007 0.022 0.000 0.007 0.001 0.000 0.002
    #     ko        0.677 0.203 0.078 0.016 0.012 0.004 0.003 0.003 0.000 0.003 0.001 0.000
    #     de        0.631 0.264 0.025 0.014 0.008 0.007 0.010 0.001 0.033 0.003 0.004 0.001
    #     it        0.798 0.117 0.014 0.006 0.037 0.008 0.012 0.002 0.004 0.004 0.000 0.000
    #     ru        0.717 0.193 0.036 0.013 0.018 0.003 0.015 0.003 0.003 0.000 0.000 0.000
    #     ja        0.627 0.276 0.049 0.018 0.009 0.009 0.004 0.009 0.000 0.000 0.000 0.000
    #     pt        0.742 0.171 0.075 0.000 0.000 0.000 0.004 0.000 0.000 0.004 0.000 0.004
    #     sv        0.639 0.270 0.057 0.000 0.000 0.000 0.016 0.000 0.000 0.016 0.000 0.000
    #     nl        0.639 0.196 0.062 0.031 0.010 0.000 0.000 0.000 0.021 0.041 0.000 0.000
    #     tr        0.688 0.219 0.062 0.016 0.016 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    #     da        0.672 0.241 0.034 0.000 0.017 0.000 0.034 0.000 0.000 0.000 0.000 0.000
    #     pl        0.759 0.111 0.037 0.019 0.019 0.000 0.037 0.000 0.000 0.000 0.000 0.019
    #     no        0.567 0.333 0.033 0.000 0.000 0.067 0.000 0.000 0.000 0.000 0.000 0.000
    #     cs        0.719 0.188 0.062 0.000 0.000 0.000 0.031 0.000 0.000 0.000 0.000 0.000
    #     el        0.750 0.083 0.042 0.042 0.000 0.000 0.083 0.000 0.000 0.000 0.000 0.000
    #     th        0.833 0.125 0.042 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    #     hu        0.778 0.167 0.056 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    #     id        1.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    #     fi        0.500 0.357 0.071 0.071 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    #     ca        0.600 0.400 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    #     is        0.600 0.200 0.200 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    #     hr        1.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    
    yp.booked = matrix(c(config$yp.te[2:12] / sum(config$yp.te[2:12])), byrow = T, ncol = 11, nrow = length(preds))
    yp.booked[langs == 'fr', ] = matrix(c(0.217, 0.016, 0.059, 0.009, 0.010, 0.009, 0.006, 0.007, 0.003, 0.002, 0.002) / 0.340, byrow = T, ncol = 11, nrow = sum(langs == 'fr'))
    yp.booked[langs == 'es', ] = matrix(c(0.181, 0.051, 0.020, 0.010, 0.007, 0.022, 0.000, 0.007, 0.001, 0.000, 0.002) / 0.301, byrow = T, ncol = 11, nrow = sum(langs == 'es'))
    yp.booked[langs == 'de', ] = matrix(c(0.264, 0.025, 0.014, 0.008, 0.007, 0.010, 0.001, 0.033, 0.003, 0.004, 0.001) / 0.370, byrow = T, ncol = 11, nrow = sum(langs == 'de'))
    yp.booked[langs == 'it', ] = matrix(c(0.117, 0.014, 0.006, 0.037, 0.008, 0.012, 0.002, 0.004, 0.004, 0.000, 0.000) / 0.204, byrow = T, ncol = 11, nrow = sum(langs == 'it'))
    yp.booked[langs == 'ru', ] = matrix(c(0.193, 0.036, 0.013, 0.018, 0.003, 0.015, 0.003, 0.003, 0.000, 0.000, 0.000) / 0.284, byrow = T, ncol = 11, nrow = sum(langs == 'ru'))
    yp.booked[langs == 'no', ] = matrix(c(0.333, 0.033, 0.000, 0.000, 0.067, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000) / 0.433, byrow = T, ncol = 11, nrow = sum(langs == 'no'))

    p = (0.97 * preds) * yp.booked # 0.93 worked best for validset, Pub LB: 0.953 => 0.88082, 0.95 => 0.88072, 0.956 => 0.88083, 0.96 => 0.88088, 0.965 => 0.88093, 0.98 => 0.88075
    p = cbind(1 - rowSums(p), p)
    p = t(apply(p, 1, order, decreasing = T))[, 1:5]
    preds = matrix(config$label.coding[c(p)], dim(p))
    return (preds)
  }
  
  if (config$careful.expand1) {
    # Always put NDF and US as either first or second slot, sort the others according to model
    p = matrix(preds, ncol = config$nr.classes, byrow = T)
    first  = ifelse(p[, 1] > config$two.class.thresh, 'NDF', 'US')
    second = ifelse(p[, 1] > config$two.class.thresh, 'US', 'NDF')
    p[, 1:2] = 0
    p345 = t(apply(p, 1, order, decreasing = T))[, 1:3]
    preds345 = matrix(config$label.coding[c(p345)], dim(p345))
    return (cbind(first, second, preds345))
  }

  if (config$careful.expand2) {
    # Always put NDF as either first or second slot, sort the others according to model
    p = matrix(preds, ncol = config$nr.classes, byrow = T)
    p[, -1] = p[, -1] * config$reweight.lambda
    p[, 1] = 1 - rowSums(p[, -1])
    p.max = apply(p, 1, max)
    idx = (p[, 1] != p.max)
    p[idx, 1] = p.max[idx] - 1e-10
    preds = t(apply(p, 1, order, decreasing = T))[, 1:5]
    preds = matrix(config$label.coding[c(preds)], dim(preds))
    return (preds)
  }

  if (config$careful.expand3) {
    # Almost always put NDF and US as either first or second slot, sort the others according to model
    p = matrix(preds, ncol = config$nr.classes, byrow = T)
    
    preds.full = t(apply(p, 1, order, decreasing = T))[, 1:5]
    preds.full = matrix(config$label.coding[c(preds.full)], dim(preds.full))
    idx = apply(p > 0.6, 1, any)
    
    first  = ifelse(p[, 1] > config$two.class.thresh, 'NDF', 'US')
    second = ifelse(p[, 1] > config$two.class.thresh, 'US', 'NDF')
    p[, 1:2] = 0
    p345 = t(apply(p, 1, order, decreasing = T))[, 1:3]
    preds345 = matrix(config$label.coding[c(p345)], dim(p345))
    preds.careful = cbind(first, second, preds345)
    
    preds = preds.careful
    preds[idx, ] = preds.full[idx, ]
    return (preds)
  }

  if (config$main.approach %in% c('NDF/booked', 'US/notUS|booked')) {
    p = cbind(1 - preds, preds)
  } else {
    p = matrix(preds, ncol = config$nr.classes, byrow = T)
  }

  if (config$main.approach == 'US/notUS|booked') {
    # for now, just expand the nont NDF class
    p.rest = config$yp.tr[3:12] / sum(config$yp.tr[3:12]) # TODO improve this a little with a user-language model?
    if (is.null(s1.preds)) {
      p = cbind(0, p[, 1], p[, 2] %o% p.rest)
    } else {
      p = cbind(1 - s1.preds, s1.preds * p[, 1], (s1.preds * p[, 2]) %o% p.rest)
    }
  } else if (config$main.approach == 'destination|booked') {
    if (is.null(s1.preds)) {
      p = cbind(0, p)
    } else {
      p = cbind(1 - s1.preds, s1.preds * p)
    }
  } else if (config$main.approach == 'destination|bookedNotUS') {
    if (is.null(s1.preds)) {
      p = cbind(0, 0, p)
    } else {
      p = cbind(1 - s1.preds, s1.preds, p - 1) # NOTE: this isn't a distribution, only a relative score per class
    }
  } else if (config$main.approach == 'other/FR/IT|oneOfThese') {
    if (is.null(s1.preds)) {
      p = cbind(0, 0, p, rep(0, 7))
    } else {
      p = cbind(1 - s1.preds, s1.preds, p - 1, (-3):(-9)) # NOTE: this isn't a distribution, only a relative score per class
    }
  } else if (!(config$main.approach %in% c('allclass', 'top5'))) {
    # We model the first config$nr.classes - 1 labels as different classes, and merge the remaining 
    # ones. Assumption: we have zero information in the features to decide between the merged 
    # labels (so we can only use the marginal testset distribution to guide us).

    nr.all.classes = length(config$label.coding)
    nr.mod.classes = config$nr.classes
    #if (config$post.reweight) {
      yp.tr.mod = config$yp.tr[nr.mod.classes:nr.all.classes] / sum(config$yp.tr[nr.mod.classes:nr.all.classes])
    #} else {
    #  yp.tr.mod = config$yp.te[nr.mod.classes:nr.all.classes] / sum(config$yp.te[nr.mod.classes:nr.all.classes])
    #}

    p.mod = p[, 1:(nr.mod.classes - 1)]
    p.unm = p[, nr.mod.classes] %o% yp.tr.mod
    p = cbind(p.mod, p.unm)
  }

  if (config$careful.expand4) {
    # Always put NDF and US as either first or second slot, sort the others according to model
    first  = ifelse(p[, 1] > config$two.class.thresh, 'NDF', 'US')
    second = ifelse(p[, 1] > config$two.class.thresh, 'US', 'NDF')
    p[, 1:2] = 0
    p345 = t(apply(p, 1, order, decreasing = T))[, 1:3]
    preds345 = matrix(config$label.coding[c(p345)], dim(p345))
    return (cbind(first, second, preds345))
  }
  
  if (config$post.reweight) {
    # Instead of simple MAP, adjust for the discrepancy between the train/test class distributions
    # We don't want to overdo it, becuase some of this information is expected to already be 
    # captured by the model through a different feature distribution, so we shrink towards 1. The
    # shrinkage can be oprimized on the validation set (and hopefully we won't overfit... this seems
    # promising, since it looks like this helps a little and it isn't sensitive to the percise
    # shinkage value)
    
    #p[, -1] = p[, -1] * config$reweight.lambda
    #p[, 1] = 1 - rowSums(p[, -1])
    
    #p = t(t(p) * (1 - config$reweight.lambda + config$reweight.lambda * config$reweight.v))
    #p = t(t(p * (1 - config$reweight.lambda)) + config$reweight.lambda * config$yp.te)
    
    # maybe this isn't accurate enough
    #adjst = c(1.15, 0.8, rep(0.7, 10))
    #p = t(t(p) * (1 - config$reweight.lambda + config$reweight.lambda * adjst))
    
    # Constrain the marginal to the one observed on the public LB
    p = p * config$yp.te / colMeans(p)
  }

  # Extract the 5 most probable classes according to the model
  # (this is the prediction that maximizes the expected gain)
  p = t(apply(p, 1, order, decreasing = T))[, 1:5]
  preds = matrix(config$label.coding[c(p)], dim(p))
  return (preds)
}

config$fixup.non.en.speakers = function(config, preds, langs) {
  for (i in which((langs == 'zh'))) { j = which(!(preds[i, ] %in% c('NDF', 'US'))); preds[i, j] = c('other', 'FR', 'IT', 'GB')[1:length(j)] }
  for (i in which((langs == 'fr'))) { j = which(!(preds[i, ] %in% c('NDF', 'US'))); preds[i, j] = c('FR', 'other', 'IT', 'GB')[1:length(j)] }
  for (i in which((langs == 'es'))) { j = which(!(preds[i, ] %in% c('NDF', 'US'))); preds[i, j] = c('other', 'ES', 'FR', 'IT')[1:length(j)] }
  for (i in which((langs == 'ko'))) { j = which(!(preds[i, ] %in% c('NDF', 'US'))); preds[i, j] = c('other', 'FR', 'IT', 'GB')[1:length(j)] }
  for (i in which((langs == 'de'))) { j = which(!(preds[i, ] %in% c('NDF', 'US'))); preds[i, j] = c('DE', 'other', 'FR', 'IT')[1:length(j)] }
  for (i in which((langs == 'it'))) { j = which(!(preds[i, ] %in% c('NDF', 'US'))); preds[i, j] = c('IT', 'other', 'ES', 'FR')[1:length(j)] }
  return (preds)
}

config$three.stage.expand = function(config, s1.preds, s2.preds, s3.preds, langs) {
  # Since this is still experimental, there will be soem code duplication with the regular version

  if (config$naive.expand1) {
    n = length(s1.preds)
    
    # By default:
    c1 = rep('NDF'  , n)
    c2 = rep('US'   , n)
    c3 = rep('other', n)
    c4 = rep('FR'   , n)
    c5 = rep('IT'   , n)
  
    # Stage1
    # Move NDF down when confident enough that the user booked
    s1.idx2 = s1.preds > 0.6 # NDF goes second
    s1.idx3 = s1.preds > 0.89 # NDF goes third
    s1.idx5 = s1.preds > 0.95 # NDF goes fifth

    s1            = cbind(c1, c2, c3, c4, c5)
    s1[s1.idx2, ] = cbind(c2, c1, c3, c4, c5)[s1.idx2, ]
    s1[s1.idx3, ] = cbind(c2, c3, c1, c4, c5)[s1.idx3, ]
    #s1[s1.idx5, ] = cbind(c2, c3, c4, c5, c1)[s1.idx5, ]
    
    # Stage 2
    # Move US down when confident enough that the user did not book domestically
    s2.idx = s2.preds > 0.8
    # FIXME add more levels? (in stage 3 too)
    c2[s2.idx] = 'other'
    c3[s2.idx] = 'US'
    
    s2            = cbind(c1, c2, c3, c4, c5)
    s2[s1.idx2, ] = cbind(c2, c1, c3, c4, c5)[s1.idx2, ]
    s2[s1.idx3, ] = cbind(c2, c3, c1, c4, c5)[s1.idx3, ]
    #s2[s1.idx5, ] = cbind(c2, c3, c4, c5, c1)[s1.idx5, ]

    # Stage 3
    # After NDF/US have been placed, populate by the highest probability remaining classes
    if (0) {    
      # A principled model
      s3.p = t(apply(matrix(s3.preds, ncol = 10, byrow = T), 1, order, decreasing = T))[, 1:3]
      s3.top3 = matrix((config$label.coding[-(1:2)])[c(s3.p)], dim(s3.p))
      c3 = s3.top3[, 1]
      c4 = s3.top3[, 2]
      c5 = s3.top3[, 3]
    } else {
      # The naive language model
      c3 = rep('other', length(langs))
      c3[langs == 'fr'] = 'FR'
      c3[langs == 'de'] = 'DE'
      c3[langs == 'it'] = 'IT'
      c3[langs == 'no'] = 'GB'
      c4 = rep('FR', length(langs))
      c4[langs == 'fr'] = 'other'
      c4[langs == 'es'] = 'ES'
      c4[langs == 'de'] = 'other'
      c4[langs == 'it'] = 'other'
      c4[langs == 'ru'] = 'IT'
      c4[langs == 'no'] = 'other'
      c5 = rep('IT', length(langs))
      c5[langs == 'de'] = 'ES'
      c5[langs == 'it'] = 'ES'
      c5[langs == 'es'] = 'FR'
      c5[langs == 'ru'] = 'ES'
      c5[langs == 'no'] = 'FR'
    }

    c2 = rep('US', n)
    c2[s2.idx] = c3[s2.idx]
    c3[s2.idx] = 'US'
    
    s3            = cbind(c1, c2, c3, c4, c5)
    s3[s1.idx2, ] = cbind(c2, c1, c3, c4, c5)[s1.idx2, ]
    s3[s1.idx3, ] = cbind(c2, c3, c1, c4, c5)[s1.idx3, ]
    #s3[s1.idx5, ] = cbind(c2, c3, c4, c5, c1)[s1.idx5, ]
  } else {
    s1.preds = cbind(1 - s1.preds, s1.preds)
    s2.preds = cbind(1 - s2.preds, s2.preds)
    #s3.preds = matrix(s3.preds, ncol = 10, byrow = T)
    s3.preds = config$three.stage.gamma * cbind(matrix(s3.preds, ncol = 3, byrow = T), 0, 0, 0, 0, 0, 0, 0)
    
    yp.booked = config$yp.te[2:12] / sum(config$yp.te[2:12])
    yp.booked[7:11] = 0
    s1 = cbind(s1.preds[, 1], s1.preds[, 2] %o% yp.booked)
  
    yp.foreign = config$yp.te[3:12] / sum(config$yp.te[3:12])
    yp.booked[7:11] = 0
    s2 = cbind(s1.preds[, 1], s1.preds[, 2] * s2.preds[, 1], (s1.preds[, 2] * s2.preds[, 2]) %o% yp.foreign)

    s3 = cbind(s1.preds[, 1], s1.preds[, 2] * s2.preds[, 1], (s1.preds[, 2] * s2.preds[, 2]) * s3.preds)
    
    if (0) {
      # Calibrate by way of the public LB marginal class distribution
      if (1) {
        # Bugged version! This seems pretty meaningless, and yet improved my public LB score substantially?!?
        s1 = s1 * config$yp.te / colMeans(s1)
        s2 = s2 * config$yp.te / colMeans(s2)
        s3 = s3 * config$yp.te / colMeans(s3)
        # This is what it does on average:
        #w = config$yp.te / colMeans(s3)
        #cbind(w, w[c(9:12, 1:8)], w[c(5:12, 1:4)]) / w
        # Maybe we can learn from this though?
        #w = c(0.98, rep(1, 11)) # => 0.88118
        #w = c(0.99, rep(1, 11)) # => 0.88119, which is the same with or without language based postprocessing
        #w = rep(c(1, 1.01, 0.99), 4) # => 0.88122
        #w = rep(c(1, 1.01, 1.01), 4) # => 0.88125, which is also what I got from w = rep(c(1, mean(w[c(9:12, 1:8)] / w), mean(w[c(5:12, 1:4)] / w)), 4)
        #w = c(1, 0.995, rep(1, 10)) # => was much worse
        #cat('Bugged calibration:', w, '\n')
        #s1 = t(t(s1) * w)
        #s2 = t(t(s2) * w)
        #s3 = t(t(s3) * w)
      } else {
        cat('Final calibration:', (config$yp.te / colMeans(s3)) / (config$yp.te[1] / colMeans(s3)[1]), '\n')
        s1 = t(t(s1) * config$yp.te / colMeans(s1))
        s2 = t(t(s2) * config$yp.te / colMeans(s2))
        s3 = t(t(s3) * config$yp.te / colMeans(s3))
      }
    } else {
      s1 = t(t(s1) * config$post.w)
      s2 = t(t(s2) * config$post.w)
      s3 = t(t(s3) * config$post.w)
    }
    
    s1 = t(apply(s1, 1, order, decreasing = T))[, 1:5]
    s2 = t(apply(s2, 1, order, decreasing = T))[, 1:5]
    s3 = t(apply(s3, 1, order, decreasing = T))[, 1:5]

    s1 = matrix(config$label.coding[c(s1)], dim(s1))
    s2 = matrix(config$label.coding[c(s2)], dim(s2))
    s3 = matrix(config$label.coding[c(s3)], dim(s3))
    
    if (1) {
      # Prioritize prominent native tounge countries where applicable
      s1 = config$fixup.non.en.speakers(config, s1, langs)
      s2 = config$fixup.non.en.speakers(config, s2, langs)
      s3 = config$fixup.non.en.speakers(config, s3, langs)
    }
  }
  
  return (list(s1 = s1, s2 = s2, s3 = s3))
}

config$error.table = function(config, preds, labels) {
  # Given expanded predictions and labels (i.e., string or factors with destination names),
  # compute the number of errors of each kind
  
  labels = factor(labels, config$label.coding)
  preds = data.frame(preds)
  for (i in 1:ncol(preds)) {
    preds[, i] = factor(preds[, i], config$label.coding)
  }
  
  t0 = table(labels)
  t1 = table(labels, preds[, 1])
  t2 = table(labels, preds[, 2])
  t3 = table(labels, preds[, 3])
  t4 = table(labels, preds[, 4])
  t5 = table(labels, preds[, 5])
  
  res = matrix(NA, 12, 7)
  colnames(res) = c('P', 1:5, '5+')
  rownames(res) = config$label.coding
  res[, 1] = config$yp.te
  res[, 2] = diag(t1)
  res[, 3] = diag(t2)
  res[, 4] = diag(t3)
  res[, 5] = diag(t4)
  res[, 6] = diag(t5)
  res[, 7] = t0 - rowSums(res[, 2:6])
  res[, 2:7] = res[, 2:7] / c(t0)
  
  res = round(res * 1000) / 1000
  
  return (res)
}

# The evaluation metric used in the competition
config$ndcg5 = function(preds, labels) {
  # preds is a (n x 5) matrix, and labels is a (n) vector
  succ = (preds == labels)
  w = 1 / log((1:5) + 1, base = 2)
  ndcgs = succ %*% w
  ndcg.mean = mean(ndcgs)
  ndcg.se = sd(ndcgs) / sqrt(length(ndcgs))
  return (list(mean = ndcg.mean, se = ndcg.se))
}

config$ndcg5.core = function(preds, labels, weights) {
  # preds is a (n x 5) matrix, and labels and weights are (n) vectors
  succ = (preds == labels)
  w = 1 / log((1:5) + 1, base = 2)
  return (sum((succ %*% w) * weights) / sum(weights))
}

# Version for XGB when all classes are coded with distinct labels
config$ndcg5.allclass = function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  weights = getinfo(dtrain, 'weight')
  pred = t(matrix(preds, ncol = length(labels)))
  pred = t(apply(pred, 1, order, decreasing = T))[, 1:5] - 1
  succ = (pred == labels)
  w = 1 / log((1:5) + 1, base = 2)
  ndcg = sum((succ %*% w) * weights) / sum(weights)
  return (list(metric = 'ndcg5', value = ndcg))
}

config$cross.validate = function(config) {
  finalized.data = config$finalize.data(config)
  
  ndcg5.someclass = function(preds, dtrain) {
    labels = getinfo(dtrain, 'label')
    weights = getinfo(dtrain, 'weight')
    
    expanded.preds = config$expand.pred(config, preds, finalized.data$ancillary$train[train.on == T, language])
    labels = finalized.data$ancillary$train[train.on == T, country_destination]
    ndcg5 = config$ndcg5.core(expanded.preds, labels, weights)
    
    return (list(metric = 'ndcg5', value = ndcg5))
  }
  
  if (config$use.precise.eval) {
    if (config$main.approach == 'allclass') {
      xgb.eval_metric = config$ndcg5.allclass
    } else {
      xgb.eval_metric = ndcg5.someclass
    }
    xgb.maximize = T
  } else {
    if (config$main.approach %in% c('NDF/booked', 'US/notUS|booked')) {
      xgb.eval_metric = 'logloss'
    } else {
      xgb.eval_metric = 'mlogloss'
    }
    xgb.maximize = F
  }
  
  if (config$main.approach %in% c('NDF/booked', 'US/notUS|booked')) {
    xgb.objective = 'binary:logistic'
  } else {
    xgb.objective = 'multi:softprob'
  }

  # These must not appear when not wanted, otherwise xgb gets agitated  
  params = list()
  if (config$nr.classes > 2) params = c(params, num_class = config$nr.classes)
  if (!is.null(config$xgb.num_parallel_tree)) params = c(params, num_parallel_tree = config$xgb.num_parallel_tree)
  
  xgb = xgb.cv(params = params,
    nfold             = 5, # class-stratified by default (but doesn't know about 2014-only session data availability and time drift, or the wierd discrepancy between test and train marginal class fistributions)
    data              = finalized.data$trainset, 
    objective         = xgb.objective,
    eval_metric       = xgb.eval_metric,
    maximize          = xgb.maximize,
    nrounds           = config$xgb.params$max.xgb.nrounds, 
    early.stop.round  = config$xgb.params$xgb.early.stop.round,
    print.every.n     = config$xgb.params$xgb.print.every.n,
    eta               = config$xgb.params$xgb.eta,
    min_child_weight  = config$xgb.params$xgb.min.child.weight,
    max_depth         = config$xgb.params$xgb.max_depth, 
    subsample         = config$xgb.params$xgb.subsample,
    colsample_bytree = config$xgb.params$xgb.colsample_bytree,
    seed              = round(runif(1) * 1e6),
    nthread           = ifelse(config$compute.backend != 'serial', 1, config$nr.threads)
  )

  cat('\n')
  
  # Hackish...
  names(xgb) = c('train.mean', 'train.std', 'test.mean', 'test.std')
  idx = which.max(xgb$test.mean)
  cv = xgb$test.mean[idx]
  cv.lo = xgb$test.mean[idx] - 2 * xgb$test.std[idx]
  cv.hi = xgb$test.mean[idx] + 2 * xgb$test.std[idx]
  
  #if (config$compute.backend != 'condor') {
  #  nn = length(xgb$test.mean)
  #  plot(1:nn, xgb$train.mean, type = 'l', lty = 2, ylab = paste('CV', xgb.eval_metric), xlab = 'Boosting round', ylim = c(cv.lo, cv.hi))
  #  lines(1:nn, xgb$test.mean + 2 * xgb$test.std, lty = 3)
  #  lines(1:nn, xgb$test.mean)
  #  lines(1:nn, xgb$test.mean - 2 * xgb$test.std, lty = 3)
  #}
  
  return (list(best.score = cv, best.round = idx))
}

config$train = function(config) {
  finalized.data = config$finalize.data(config)

  if (is.null(finalized.data$validset)) {
    watchlist = list(train = finalized.data$trainset)
  } else {
    vl = getinfo(finalized.data$validset, 'label')
    watchlist = list(test = slice(finalized.data$validset, which(vl >= 0)), train = finalized.data$trainset)
  }

  # Version for XGB when only some of the classes are coded with distinct labels
  # FIXME this is actually VERY slow! Optimize by precomputing the score per sample for each prediction?
  ndcg5.someclass = function(preds, dtrain) {
    labels = getinfo(dtrain, 'label')
    weights = getinfo(dtrain, 'weight')
    
    # Unfortunately, XGB doesn't allow storing custom ancillary information in its data matrix format.
    # So (although customizing xgb is easy) I'll improvise...
    if (length(labels) == nrow(finalized.data$validset)) { # this is actually the validation set
      expanded.preds = config$expand.pred(config, preds, finalized.data$ancillary$train[valid.on == T, language])
      labels = finalized.data$ancillary$train[valid.on == T, country_destination]
    } else {
      expanded.preds = config$expand.pred(config, preds, finalized.data$ancillary$train[train.on == T, language])
      labels = finalized.data$ancillary$train[train.on == T, country_destination]
    }
    ndcg5 = config$ndcg5.core(expanded.preds, labels, weights)
    
    return (list(metric = 'ndcg5', value = ndcg5))
  }
  
  if (config$use.precise.eval) {
    if (config$main.approach == 'allclass') {
      xgb.eval_metric = ndcg5.someclass #config$ndcg5.allclass
    } else {
      xgb.eval_metric = ndcg5.someclass
    }
    xgb.maximize = T
  } else {
    if (config$main.approach %in% c('NDF/booked', 'US/notUS|booked')) {
      xgb.eval_metric = 'logloss'
    } else {
      xgb.eval_metric = 'mlogloss'
    }
    xgb.maximize = F
  }

  if (config$main.approach %in% c('NDF/booked', 'US/notUS|booked')) {
    xgb.objective = 'binary:logistic'
  } else {
    xgb.objective = 'multi:softprob'
  }

  # These must not appear when not wanted, otherwise xgb gets agitated  
  params = list()
  if (config$nr.classes > 2) params = c(params, num_class = config$nr.classes)
  if (!is.null(config$xgb.num_parallel_tree)) params = c(params, num_parallel_tree = config$xgb.num_parallel_tree)
  
  xgb = xgb.train(params = params,
    data              = finalized.data$trainset, 
    objective         = xgb.objective,
    eval_metric       = xgb.eval_metric,
    maximize          = xgb.maximize,
    watchlist         = watchlist,
    nrounds           = config$xgb.params$xgb.nrounds, 
    early.stop.round  = config$xgb.params$xgb.early.stop.round,
    print.every.n     = config$xgb.params$xgb.print.every.n,
    eta               = config$xgb.params$xgb.eta,
    min_child_weight  = config$xgb.params$xgb.min.child.weight,
    max_depth         = config$xgb.params$xgb.max_depth, 
    subsample         = config$xgb.params$xgb.subsample,
    colsample_bytree  = config$xgb.params$xgb.colsample_bytree,
    seed              = round(runif(1) * 1e6),
    nthread           = ifelse(config$compute.backend != 'serial', 1, config$nr.threads)
  )

  if (length(watchlist) == 2 && !is.null(config$xgb.params$xgb.early.stop.round)) {
    cat('\n', date(), 'Best validation result:', xgb$bestScore, '@ round', xgb$bestInd, '\n')
  }

  # Generate predictions and save them
  testset  = xgb.DMatrix(paste0(config$tmp.dir, '/xgb-testset-' , config$data.tag, '.data'))
  test.preds = predict(xgb, testset)

  if (config$do.manual.cv1 || config$validation.scheme != 'none') {
    valid.preds = predict(xgb, finalized.data$validset)
    valid.ancillary = finalized.data$ancillary$train[valid.on == T]
    if (config$validation.scheme != 'none') {
      preds = config$expand.pred(config, valid.preds, valid.ancillary$language)
      score = config$ndcg5(preds, valid.ancillary$country_destination)
      cat('Validation score:', score$mean, '+', score$se, '\n')
    }
  } else {
    valid.preds = NULL
    valid.ancillary = NULL
  }
  
  if (config$do.importance) {
    # Examine variable importance (takes some time to extract!)
    cat(date(), 'Examining importance of features in the single XGB model\n')
    impo = xgb.importance(finalized.data$ancillary$feature.names, model = xgb)
    print(impo[1:min(100, length(finalized.data$ancillary$feature.names)), ])
    save(impo, file = paste0(config$tmp.dir, '/feature-importance-', config$model.tag, '.RData'))
  }

  flnm = paste0(config$tmp.dir, '/preds-', config$model.tag, '.RData')
  save(valid.preds, test.preds, valid.ancillary, file = flnm)
}

config$manual.cross.validate1 = function(config) {
  orig.model.tag = config$model.tag
  for (i in 1:5) {
    cat(date(), 'Working on fold', i, '\n')
    config$manual.cv.fold = i
    config$model.tag = paste0(orig.model.tag, '-fold', i)
    config$train(config) # saves validation-fold and testset predictions
  }
}

config$manual.cross.validate2 = function(config) {
  # Collect the data from the different stages and folds
  for (m in 1:3) {
    stackdat.m = NULL
    for (i in 1:5) {
      load(paste0(config$tmp.dir, '/preds-', m, '-fold', i, '.RData')) # => valid.preds, test.preds, valid.ancillary
      stackdat0 = valid.ancillary[, .(id, country_destination, language)]
      if (m == 3) {
        valid.preds = matrix(valid.preds, ncol = 3, byrow = T)
        colnames(valid.preds) = paste0('z', m, '.', 1:ncol(valid.preds))
      } else {
        valid.preds = matrix(valid.preds, ncol = 1 , byrow = T)
        colnames(valid.preds) = paste0('z', m)
      }
      stackdat0 = cbind(stackdat0, valid.preds)
      stackdat.m = rbind(stackdat.m, stackdat0)
    }
    
    if (m == 1) {
      stackdat = stackdat.m
    } else {
      stopifnot(all(stackdat.m$id == stackdat$id))
      stackdat = cbind(stackdat, stackdat.m[, 4:ncol(stackdat.m), with = F])
    }
  }

  if (0) {
    # Suppose we know the marginal distribution of the testset (in practice we only have an estimate
    # from a ~20K sample iid (?) data... but this doesn't seem to matter too much)
    config$yp.te = table(factor(stackdat$country_destination, levels = config$label.coding)) / nrow(stackdat)
  } else {
    # Bootstrap to match the public LB class distribution
    y = as.numeric(factor(stackdat$country_destination, levels = config$label.coding))
    idx = sample(nrow(stackdat), nrow(stackdat), replace = T, prob = (config$yp.te / table(y))[y])
    #cbind(config$yp.te, sort(table(stackdat$country_destination[idx]), decreasing = T) / nrow(stackdat))
    stackdat = stackdat[idx]
  }
  
  # Now we can compare different ways of mapping the predictions to NDCG5 scores
  # I will try a grid of multiplicative factors applied to the probabilities of NDF and US
  
  if (0) {
    w12.grid = expand.grid(seq(0.9, 1.1, len = 11), seq(0.9, 1.1, len = 11))
    
    mapping.obj = function(w12) {
      config$post.w = c(unlist(w12), rep(1, 10))
      cv.top5.preds = config$three.stage.expand(config, stackdat$z1, stackdat$z2, c(t(data.matrix(stackdat[, 5 + (1:3), with = F]))), stackdat$language)
      s2.ndcg5 = config$ndcg5(cv.top5.preds$s2, stackdat$country_destination)
      return (s2.ndcg5$mean)
    }
  
    res = apply(w12.grid, 1, mapping.obj)
    cat('Best NDCG5:', max(res), 'at w1 =', w12.grid$Var1[which.max(res)], 'w2 =', w12.grid$Var2[which.max(res)], '\n')
    contourplot(ndcg5 ~ Var1 + Var2, cbind(ndcg5 = res, w12.grid))
    config$post.w = c(unlist(w12.grid[which.max(res), ]), rep(1, 10))
  }
  
  if (1) {
    mapping.obj = function(gamma) {
      config$three.stage.gamma = gamma
      cv.top5.preds = config$three.stage.expand(config, stackdat$z1, stackdat$z2, c(t(data.matrix(stackdat[, 5 + (1:3), with = F]))), stackdat$language)
      s3.ndcg5 = config$ndcg5(cv.top5.preds$s3, stackdat$country_destination)
      return (s3.ndcg5$mean)
    }
    
    gamma.grid = seq(0.2, 0.7, len = 100)
    res = sapply(gamma.grid, mapping.obj)
    cat('Best NDCG5:', max(res), 'at gamma =', gamma.grid[which.max(res)], '\n')
    plot(gamma.grid, res)
    config$three.stage.gamma = gamma.grid[which.max(res)]
  }

  cv.top5.preds = config$three.stage.expand(config, stackdat$z1, stackdat$z2, c(t(data.matrix(stackdat[, 5 + (1:3), with = F]))), stackdat$language)
  cv.top5.preds$s0 = matrix(c('NDF', 'US', 'other', 'FR', 'IT'), nrow = nrow(stackdat), ncol = 5, byrow = T)
  
  if (0) {
    print(config$error.table(config, cv.top5.preds$s0, stackdat$country_destination))
    print(config$error.table(config, cv.top5.preds$s1, stackdat$country_destination))
    print(config$error.table(config, cv.top5.preds$s2, stackdat$country_destination))
    print(config$error.table(config, cv.top5.preds$s3, stackdat$country_destination))
  }
  
  s0.ndcg5 = config$ndcg5(cv.top5.preds$s0, stackdat$country_destination)
  s1.ndcg5 = config$ndcg5(cv.top5.preds$s1, stackdat$country_destination)
  s2.ndcg5 = config$ndcg5(cv.top5.preds$s2, stackdat$country_destination)
  s3.ndcg5 = config$ndcg5(cv.top5.preds$s3, stackdat$country_destination)
  
  cat('\n')
  cat('CV NDCG5:\n')
  cat('Stage0:', s0.ndcg5$mean, '+', s0.ndcg5$se, '\n')
  cat('Stage1:', s1.ndcg5$mean, '+', s1.ndcg5$se, '\n')
  cat('Stage2:', s2.ndcg5$mean, '+', s2.ndcg5$se, '\n')
  cat('Stage3:', s3.ndcg5$mean, '+', s3.ndcg5$se, '\n')
  cat('\n')
}

config$postprocess = function(config) { 
  load(file = paste0(config$tmp.dir, '/preds-', config$model.tag, '.RData')) # => valid.preds, test.preds
  load(file = paste0(config$tmp.dir, '/pp-data-ancillary-', config$data.tag, '.RData')) # => ancillary

  if (0) {
    # Optimize two class thresh for naive expansion of the ndf/not approach
    # FIXME this can only be done if we have a validset, and then the valid error is optimistic
    nr.t = 50
    tgrid = seq(0.4, 0.6, len = nr.t)
    #tgrid = seq(0.55, 0.65, len = nr.t)
    #tgrid = seq(0.3, 0.5, len = nr.t)
    ndcg5.grid = rep(NA, nr.t)
    for (ti in 1:nr.t) {
      config$two.class.thresh = tgrid[ti]
      vp = config$expand.pred(config, valid.preds, ancillary$va.langs)
      ndcg5.grid[ti] = config$ndcg5.core(vp, ancillary$va.labels, ancillary$va.weights)
    }
    plot(tgrid, ndcg5.grid)
    config$two.class.thresh = tgrid[which.max(ndcg5.grid)]
    cat('Best naive two class thresh is', config$two.class.thresh, '\n')
  }

  if (0) {
    # Optimize lambda for adjustment of predictions to testset class distribution
    # FIXME this can only be done if we have a validset, and then the valid error is optimistic
    nr.l = 50
    #lgrid = seq(0.8, 1.2, len = nr.l)
    lgrid = seq(0.1, 2, len = nr.l)
    #lgrid = seq(0.2, 0.8, len = nr.l)
    #lgrid = seq(0, 0.2, len = nr.l)
    ndcg5.grid = rep(NA, nr.l)
    for (li in 1:nr.l) {
      config$reweight.lambda = lgrid[li]
      vp = config$expand.pred(config, valid.preds, ancillary$va.langs)
      ndcg5.grid[li] = config$ndcg5.core(vp, ancillary$va.labels, ancillary$va.weights)
    }
    plot(lgrid, ndcg5.grid)
    config$reweight.lambda = lgrid[which.max(ndcg5.grid)]
    cat('Best reweight.lambda is', config$reweight.lambda, '\n')
  }
  
  test.preds = config$expand.pred(config, test.preds, ancillary$te.langs)

  if (config$validation.scheme != 'none') {
    if (config$main.approach %in% c('US/notUS|booked', 'destination|booked', 'destination|bookedNotUS', 'other/FR/IT|oneOfThese')) {
      valid.preds = config$expand.pred(config, valid.preds, ancillary$fva.langs)
      ndcg5 = config$ndcg5.core(valid.preds, ancillary$fva.labels, ancillary$fva.weights)
    } else {
      valid.preds = config$expand.pred(config, valid.preds, ancillary$va.langs)
      ndcg5 = config$ndcg5.core(valid.preds, ancillary$va.labels, ancillary$va.weights)
    }
    cat('Final validation ndcg5 error:', ndcg5, '\n')
  } else {
    valid.preds = NULL
  }
  
  save(valid.preds, test.preds, file = paste0(config$tmp.dir, '/final-preds-', config$model.tag, '.RData'))
}

config$generate.submission = function(config) {
  load(file = paste0(config$tmp.dir, '/final-preds-', config$model.tag, '.RData')) # => valid.preds, test.preds
  load(file = paste0(config$tmp.dir, '/pp-data-ancillary-', config$data.tag, '.RData')) # => ancillary

  predictions = data.frame(id = ancillary$test$id, country = test.preds)
  
  submission = melt(predictions, 'id') # NOTE: ignore the warning about attributes
  submission = submission[order(submission$id, submission$variable), -2]
  names(submission) = c('id', 'country')
  
  write.csv(submission, paste0('submission-', config$submit.tag, '.csv'), quote = F, row.names = F)
  zip(paste0('submission-', config$submit.tag, '.zip'), paste0('submission-', config$submit.tag, '.csv'))
  
  ref.submission = read.csv('submission-ens5.csv') # my best submission so far
  ref.submission = ref.submission[seq(1, nrow(ref.submission), by = 5), ]
  cmpr = merge(predictions, ref.submission, by = 'id')
  cat('Sanity check: first prediction per user matches my best submission', mean(as.character(cmpr$country) == as.character(cmpr[, 2])), 'of the time\n')

  cat('Distribution of prediction slots:\n')
  all.pred.class.dist = NULL
  for (i in 1:5) {
    pred.class.dist = table(factor(predictions[, i + 1], levels = config$label.coding))
    all.pred.class.dist = rbind(all.pred.class.dist, pred.class.dist / sum(pred.class.dist))
  }
  print(round(all.pred.class.dist * 1000) / 1000)
  cat('\n')
}

config$complex.three.stage = function(config) {
  cat(date(), 'Complex mode: three-stage model\n')

  do.stage1 = T
  do.stage2 = T
  do.stage3 = T
  
  if (do.stage1) {
    cat(date(), 'Stage I\n')
    
    config$model.tag = 'stage1'
    config$main.approach = config$s1.main.approach
    config$nr.classes = config$s1.nr.classes
    config$xgb.params = config$s1.params
    config$xgb.params$xgb.early.stop.round  = ceiling(config$xgb.params$xgb.nrounds / 5)
    config$xgb.params$xgb.print.every.n     = ceiling(config$xgb.params$xgb.nrounds / 20)
    config.stage1 = config
    
    if (config$do.preprocess) {
      cat(date(), 'Preprocessing\n')
      config$preprocess(config)
    }
    if (config$do.train) {
      cat(date(), 'Training\n')
      config$train(config)
    }
  }
  
  if (do.stage2) {
    cat(date(), 'Stage II\n')
    
    config$model.tag = 'stage2'
    config$main.approach = config$s2.main.approach
    config$nr.classes = config$s2.nr.classes
    config$xgb.params = config$s2.params
    config$xgb.params$xgb.early.stop.round  = ceiling(config$xgb.params$xgb.nrounds / 5)
    config$xgb.params$xgb.print.every.n     = ceiling(config$xgb.params$xgb.nrounds / 20)
    config.stage2 = config
    
    if (config$do.preprocess) {
      cat(date(), 'Preprocessing\n')
      config$preprocess(config)
    }
    if (config$do.train) {
      cat(date(), 'Training\n')
      config$train(config)
    }
  }

  if (do.stage3) {
    cat(date(), 'Stage III\n')
    
    config$model.tag = 'stage3'
    config$main.approach = config$s3.main.approach
    config$nr.classes = config$s3.nr.classes
    config$xgb.params = config$s2.params
    config$xgb.params$xgb.early.stop.round  = ceiling(config$xgb.params$xgb.nrounds / 5)
    config$xgb.params$xgb.print.every.n     = ceiling(config$xgb.params$xgb.nrounds / 20)
    config.stage3 = config
    
    if (config$do.preprocess) {
      cat(date(), 'Preprocessing\n')
      config$preprocess(config)
    }
    if (config$do.train) {
      cat(date(), 'Training\n')
      config$train(config)
    }
  }
  
  #
  # Custom postprocess by connecting the models
  #

  load(file = paste0(config$tmp.dir, '/pp-data-ancillary-1.RData')) # => ancillary
  
  config$model.tag = 'threestage'
  load(file = paste0(config$tmp.dir, '/preds-stage1.RData')) # => valid.preds, test.preds
  s1.test.preds = test.preds
  s1.valid.preds = valid.preds
  load(file = paste0(config$tmp.dir, '/preds-stage2.RData')) # => valid.preds, test.preds
  s2.test.preds = test.preds
  s2.valid.preds = valid.preds
  load(file = paste0(config$tmp.dir, '/preds-stage3.RData')) # => valid.preds, test.preds
  s3.test.preds = test.preds
  s3.valid.preds = valid.preds
  
  #cat('NOTE: using only stages 1 and 2!\n')
  test.preds = config$three.stage.expand(config, s1.test.preds, s2.test.preds, s3.test.preds, ancillary$test[, language])$s3
  
  if (config$validation.scheme != 'none') {
    valid.preds.all = config$three.stage.expand(config, s1.valid.preds, s2.valid.preds, s3.valid.preds, ancillary$train[valid.on == T, language])
    s1.ndcg5 = config$ndcg5.core(valid.preds.all$s1, ancillary$train[valid.on == T, country_destination], ancillary$train[valid.on == T, weight])
    s2.ndcg5 = config$ndcg5.core(valid.preds.all$s2, ancillary$train[valid.on == T, country_destination], ancillary$train[valid.on == T, weight])
    s3.ndcg5 = config$ndcg5.core(valid.preds.all$s3, ancillary$train[valid.on == T, country_destination], ancillary$train[valid.on == T, weight])
    cat('Validation ndcg5 scores per stage:', s1.ndcg5, s2.ndcg5, s3.ndcg5, '\n')
    #cat('In stage 1:\n')
    #print(config$error.table(config, valid.preds.all$s1, ancillary$va.labels))
    #cat('In stage 2:\n')
    #print(config$error.table(config, valid.preds.all$s2, ancillary$va.labels))
    #cat('In stage 3:\n')
    #print(config$error.table(config, valid.preds.all$s3, ancillary$va.labels))
    valid.preds = valid.preds.all$s3
  }
  
  save(valid.preds, test.preds, file = paste0(config$tmp.dir, '/final-preds-', config$model.tag, '.RData'))

  cat(date(), 'Generating submission\n')
  config$generate.submission(config)
}

# Do stuff
# ==============================================================================

set.seed(config$rng.seed)

if (config$mode == 'single') {
  if (config$do.preprocess) {
    cat(date(), 'Preprocessing\n')
    config$preprocess(config)
  } else {
    cat(date(), 'NOTE: using previously preprocessed data\n')
  }
  
  if (config$do.manual.cv1) {
    cat(date(), 'Manual cross-validating 1\n')
    config$manual.cross.validate1(config)
  } else if (config$do.manual.cv2) {
    cat(date(), 'Manual cross-validating 2\n')
    config$manual.cross.validate2(config)
  } else if (config$do.cv) {
    cat(date(), 'Cross-validating\n')
    config$cross.validate(config)
  } else if (config$do.train) {
    cat(date(), 'Training\n')
    config$train(config)
  }
  
  if (config$do.postprocess) {
    cat(date(), 'Postprocessing\n')
    config$postprocess(config)
  }
  
  if (config$do.submission) {
    cat(date(), 'Generating submission\n')
    config$generate.submission(config)
  }
} else if (config$mode == 'complex') {
  if (config$complex.select == 'two-stage') {
    config$complex.two.stage(config)
  } else if (config$complex.select == 'three-stage') {
    config$complex.three.stage(config)
  } else {
    stop('Unexpected complex.select')
  }
} else if (config$mode == 'tuning') {
  cat('Tuning model\n')
  
  # 1. Define the hyperparameter grid
  if (0) {
    n.rounds = config$xgb.params$max.xgb.nrounds
    eta = c(0.003, 0.005, 0.01)
    subsample = c(0.3, 0.5, 0.8)
    colsample_bytree = c(0.3, 0.5, 0.8)
    max_depth = c(4, 6, 8, 10)
    min.child.weight = c(1, 10, 100)
  } else { # for debugging
    n.rounds = 5
    eta = 0.25
    subsample = 0.6
    colsample_bytree = 0.6
    max_depth = 2:3
    min.child.weight = 1
  }
  
  config$hp.grid = expand.grid(n.rounds = n.rounds, eta = eta, subsample = subsample, colsample_bytree = colsample_bytree, max_depth = max_depth, min.child.weight = min.child.weight)
  stopifnot(config$compute.backend != 'multicore') # this is suitable for condor or serial
  config$nr.cores = nrow(config$hp.grid)
  
  # 2. Define a function that sets up a single grid point and calls validate
  tune.job = function(config, core) {
    if (core > nrow(config$hp.grid)) return (NULL)
    
    config$model.tag = paste0('tune-', core)
    
    config$xgb.params$xgb.nrounds          = config$hp.grid$n.rounds        [core]
    config$xgb.params$xgb.eta              = config$hp.grid$eta             [core]
    config$xgb.params$xgb.max_depth        = config$hp.grid$max_depth       [core]
    config$xgb.params$xgb.subsample        = config$hp.grid$subsample       [core]
    config$xgb.params$xgb.colsample_bytree = config$hp.grid$colsample_bytree[core]
    config$xgb.params$xgb.min.child.weight = config$hp.grid$min.child.weight[core] 

    config$xgb.params$xgb.early.stop.round = ceiling(config$xgb.params$xgb.nrounds / 5)
    config$xgb.params$xgb.print.every.n    = ceiling(config$xgb.params$xgb.nrounds / 20)
    
    cat(date(), 'Current tuning setup:\n')
    print(config$hp.grid[core, ])
    cat('\n')
    
    if (config$do.cv) {
      cat(date(), 'Cross-validating\n')
      res.single = config$cross.validate(config)
    }
    
    if (config$do.train) {
      cat(date(), 'Training on all data\n')
      res.single = config$train(config)
    }
    
    return (data.frame(res.single, config$hp.grid[core, ]))
  }
  
  # 3. Call it through ComputeBackend
  cat(date(), 'Launching run\n')
  res = compute.backend.run(
    config, tune.job, combine = rbind, 
    package.dependencies = config$package.dependencies,
    source.dependencies  = config$source.dependencies,
    cluster.dependencies = config$cluster.dependencies,
    cluster.batch.name = 'dscience', 
    cluster.requirements = config$cluster.requirements
  )
  
  save(res, file = 'tuning-results.RData')
  
  cat(date(), '\nTuning results:\n\n')
  res = res[order(res$best.score), ]
  print(res)
  cat('\n')
} else if (config$mode == 'bagging') {
  if (1) {
    # For debugging
    config$nr.cores = 20
    config$xgb.params$xgb.nrounds = 50
  }
  
  bag.job = function(config, core) {
    cat(date(), 'Working on bag', core, '\n')
    config$model.tag = paste0('bag-', core)
    config$train(config)
    return (0)
  }
  
  cat(date(), 'Launching run\n')
  
  compute.backend.run(
    config, bag.job, combine = c, 
    package.dependencies = config$package.dependencies,
    source.dependencies  = config$source.dependencies,
    cluster.dependencies = config$cluster.dependencies,
    cluster.batch.name = 'dscience', 
    cluster.requirements = config$cluster.requirements
  )
  
  cat(date(), 'Merging results\n')

  merged.valid.preds = merged.test.preds = 0
  for (i in 1:(config$nr.cores)) {  
    # Load each result file, and average the predicted class probabilities, then expand.
    load(paste0(config$tmp.dir, '/preds-bag', i, '.RData')) # => valid.preds, test.preds
    merged.valid.preds = (merged.valid.preds * (i - 1) + valid.preds) / i
    merged.test.preds  = (merged.test.preds  * (i - 1) + test.preds ) / i
    
    if (config$do.postprocess) {
      cat('Intermediate bag performance follows:\n')
      valid.preds = merged.valid.preds
      test.preds = merged.test.preds
      save(valid.preds, test.preds, file = paste0(config$tmp.dir, '/preds-', config$model.tag, '.RData'))
      config$postprocess(config)
    }
  }  

  valid.preds = merged.valid.preds
  test.preds = merged.test.preds
  save(valid.preds, test.preds, file = paste0(config$tmp.dir, '/preds-', config$model.tag, '.RData'))
  
  if (config$do.postprocess) {
    cat(date(), 'Postprocessing\n')
    config$postprocess(config)
  }
  
  if (config$do.submission) {
    cat(date(), 'Generating submission\n')
    config$generate.submission(config)
  }
} else {
  stop('Unexpeced mode')
}

cat(date(), 'Done.\n')
