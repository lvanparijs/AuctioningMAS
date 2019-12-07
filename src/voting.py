import numpy

#INPUTS
k = 5 #number of sellers
n = k+1 #number of buyers
r = 5 #number of auction rounds

s_max = 5 #universal maximum price
e = 0.2 #penalty factor

lvl_commitement = True #Flag whether leveled commitment is used, otherwise its pure

#OUTPUTS
market_price = 0 #Market price

buyer_profit = numpy.zeros(n) #Profit of each buyer
buyer_paid = numpy.zeros(n) #How much each buyer paid
buyer_market_price = numpy.zeros(n)
buy_from = -numpy.ones(n) #Registers from which seller the buyer bought its item

seller_profit = numpy.zeros(k) #Profits for each seller
sellers_price = numpy.random.uniform(0,s_max,k) #Random starting price for sellers
sellers_id = numpy.arange(0,k) #Ids for each seller, used to randomise the seller order each round

bidding_factors = numpy.random.uniform(1.01,2,n) #Bidding factors
bid_inc_factors = numpy.random.uniform(1.01,2,n) #Bid increase factors, randomly generated
bid_dec_factors = numpy.random.uniform(0.01,1,n) #Bid decrease factors, randomly generated


def vickrey_auction():
    if n > k: #more buyers than sellers necessary
        for rr in range(r): #Amount of rounds
            sellers_price = numpy.random.uniform(0, s_max, k) #Generate the starting prices of each seller randomly
            numpy.random.shuffle(sellers_id) #Ensures random order at every round
            out = [] #List of which buyers are not allowed to bid anymore, user for PURE auctioning
            reset()

            print("===============")
            print("ROUND "+str(rr)) #Printing stuff
            print("===============")

            for kk in range(k): #Each seller has one auctions per round
                cur_seller_id = sellers_id[kk] #Gets the current seller ID, random due to shuffle function before each round
                start_price = sellers_price[cur_seller_id] #Get the starting price of the current seller

                bids = bidding_strategy(start_price)

                bids[out[:]] = -numpy.inf #If they have already won an auction, set bid to 0 so they dont participate
                market_price = numpy.average(bids[bids>0]) #Only averages the actual bids, not the -inf

                print("AUCTION "+str(kk)+", organised by id: "+str(cur_seller_id))
                print("Starting price: "+str(start_price))
                print("Bids: "+str(bids))
                print("Market Price: "+str(market_price))
                print("===============")

                win_id, win_price = find_nearest_less_than(market_price,bids) #Get the highest bid lower than the market value, the buyers that bid higher than market price wont go through with it since it is irrational
                #Since the winner only needs to pay the second highest price the win price needs to be adapted
                _, win_price2 = find_nearest_less_than(win_price,bids) #Find second highest price
                if win_price2 == -numpy.inf: #means there is only one valid bid below market value
                    win_price = (win_price+start_price)/2 #Average bid and starting price
                else:
                    win_price = win_price2 #Set second highest price as winning price

                print("WINNING BID: (id)"+str(win_id)+", (price)"+str(win_price))

                if lvl_commitement: #Leveled Commitment, dont remove the winner
                    print("LVL COMMITMENT AUCTIONING")
                    if buy_from[win_id] >= 0: #If a buyer previously won an auction already
                        #Settle with previous auctioner
                        print("WON PREVIOUS AUCTION!")
                        penalty_fee = e*buyer_paid[win_id] #Calculate penalty fee
                        seller_profit[int(buy_from[win_id])] -= (buyer_paid[win_id]-penalty_fee) #Seller refunds buyer, but keeps fee
                        buyer_profit[win_id] = -penalty_fee #Pay fee to seller
                        buy_from[win_id] = cur_seller_id #Change buy from to new auction

                    seller_profit[cur_seller_id] += win_price
                    buyer_paid[win_id] = win_price
                    buyer_profit[win_id] += market_price - win_price
                    buyer_market_price[win_id] = market_price
                    buy_from[win_id] = cur_seller_id
                else: #Pure Auction, remove the winner from bidders
                    print("PURE AUCTIONING")
                    out += [win_id] #Add winner to out list
                    seller_profit[cur_seller_id] += win_price
                    buyer_paid[win_id] = win_price
                    buyer_profit[win_id] += market_price - win_price
                    buyer_market_price[win_id] = market_price
                    buy_from[win_id] = cur_seller_id

                update_bidding_factors(bidding_factors,bids,market_price,out,win_id)

                print("SELLER PROFITS: " + str(seller_profit))
                print("BUYER PROFITS: " + str(buyer_profit))
                print("BUYER BIDDING FACTORS"+str(bidding_factors))
                print("BUY FROM: " + str(buy_from))
    else:
        print("INSUFFICIENT BUYERS")
        print("The number of buyers(n) must always be greater than number of sellers(k)")

def reset():
    sellers_price = numpy.random.uniform(0, s_max, k)  # Random starting price for sellers
    buyer_paid = numpy.zeros(n)  # How much each buyer paid
    buyer_market_price = numpy.zeros(n)
    bidding_factors = numpy.random.uniform(1.01, 2, n)  # Bidding factors
    bid_inc_factors = numpy.random.uniform(1.01, 2, n)  # Bid increase factors, randomly generated
    bid_dec_factors = numpy.random.uniform(0.01, 1, n)  # Bid decrease factors, randomly generated

def bidding_strategy(start_price):
    v = numpy.zeros(n)
    for i in range(n):
        if buy_from[i] != -1: #If this buyer has bought something before
            v[i] = ((buyer_market_price[i]-buyer_paid[i])+e*buyer_paid[i])
    return bidding_factors*start_price-v

def update_bidding_factors(bidding_factors, bids, market_price, out, winner_id): #Updates the bidding factors
    ids = numpy.arange(0, k)
    if len(out) > 0:
        for o in out:
            numpy.delete(ids,o)
    for id in ids:
        if id == winner_id or bids[id] >= market_price: #If this bidder won the last auction
            bidding_factors[id] *= bid_dec_factors[id]
        else:
            bidding_factors[id] *= bid_inc_factors[id]

def find_nearest_less_than(search_val, input_data):
    diff = input_data - search_val
    diff[diff>0] = -numpy.inf
    idx = diff.argmax()
    return idx, input_data[idx]


#b = [-1,2,2,2,-1,-1,-1,-1]
#print(numpy.average(b[b>0]))
#print(find_nearest_less_than(-0.1,numpy.array([0.,1.,1.4,2.])))
#print(find_nearest_less_than(-1.5,numpy.array([0.,1.,1.4,-2.])))

vickrey_auction()