#property strict
#property description "Read-only GOLD M1 data bridge for Gold Scalping AI Lab"

input string BridgeUrl = "http://127.0.0.1:8765/ingest";
input string BridgeSymbol = "GOLD";
input int SendEverySeconds = 2;
input int HistoricalBars = 500;
input int BatchSize = 100;

long last_send = 0;
bool history_sent = false;

string AccountModeText()
{
   long mode = AccountInfoInteger(ACCOUNT_TRADE_MODE);
   if(mode == ACCOUNT_TRADE_MODE_REAL) return "REAL";
   if(mode == ACCOUNT_TRADE_MODE_DEMO) return "DEMO";
   if(mode == ACCOUNT_TRADE_MODE_CONTEST) return "CONTEST";
   return "UNKNOWN";
}

string JsonEscape(string s)
{
   StringReplace(s, "\\", "\\\\");
   StringReplace(s, "\"", "\\\"");
   return s;
}

string IsoTime(datetime t)
{
   string ts = TimeToString(t, TIME_DATE|TIME_MINUTES|TIME_SECONDS);
   StringReplace(ts, ".", "-");
   StringReplace(ts, " ", "T");
   ts += "Z";
   return ts;
}

string BatchUrl()
{
   string u = BridgeUrl;
   int p = StringFind(u, "/ingest");
   if(p >= 0) return StringSubstr(u, 0, p) + "/ingest_batch";
   return u + "/ingest_batch";
}

bool PostJson(string url, string body)
{
   char data[];
   StringToCharArray(body, data, 0, WHOLE_ARRAY, CP_UTF8);
   if(ArraySize(data) > 0) ArrayResize(data, ArraySize(data)-1);

   char result[];
   string response_headers;
   string headers = "Content-Type: application/json\r\n";

   ResetLastError();
   int code = WebRequest("POST", url, headers, 5000, data, result, response_headers);
   if(code == -1)
   {
      Print("GoldBridgeEA WebRequest error: ", GetLastError(), ". Add bridge URL to MT5 allowed WebRequest URLs.");
      return false;
   }
   if(code < 200 || code >= 300)
   {
      Print("GoldBridgeEA HTTP error: ", code);
      return false;
   }
   return true;
}

string PacketJson(string symbol, MqlRates &bar, double bid, double ask, double spread)
{
   return StringFormat(
      "{\"account_id\":\"%I64d\",\"account_mode\":\"%s\",\"broker\":\"%s\",\"symbol\":\"%s\",\"timestamp\":\"%s\",\"open\":%.8f,\"high\":%.8f,\"low\":%.8f,\"close\":%.8f,\"bid\":%.8f,\"ask\":%.8f,\"spread\":%.8f,\"balance\":%.2f,\"equity\":%.2f}",
      AccountInfoInteger(ACCOUNT_LOGIN),
      AccountModeText(),
      JsonEscape(AccountInfoString(ACCOUNT_COMPANY)),
      JsonEscape(symbol),
      IsoTime(bar.time),
      bar.open,
      bar.high,
      bar.low,
      bar.close,
      bid,
      ask,
      spread,
      AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoDouble(ACCOUNT_EQUITY)
   );
}

bool SendHistory()
{
   string symbol = BridgeSymbol;
   if(!SymbolSelect(symbol, true))
   {
      Print("GoldBridgeEA cannot select symbol: ", symbol);
      return false;
   }

   int wanted = MathMax(300, HistoricalBars);
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(symbol, PERIOD_M1, 1, wanted, rates);
   if(copied < 300)
   {
      Print("GoldBridgeEA history not ready. Copied M1 bars: ", copied);
      return false;
   }

   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int chunk = MathMax(10, MathMin(BatchSize, 200));
   string url = BatchUrl();

   for(int oldest = copied - 1; oldest >= 0; )
   {
      string body = "[";
      int added = 0;

      while(oldest >= 0 && added < chunk)
      {
         MqlRates bar = rates[oldest];
         double spread = (double)bar.spread * point;
         if(spread <= 0.0)
            spread = (double)SymbolInfoInteger(symbol, SYMBOL_SPREAD) * point;

         double bid = bar.close - spread / 2.0;
         double ask = bar.close + spread / 2.0;
         if(added > 0) body += ",";
         body += PacketJson(symbol, bar, bid, ask, spread);

         oldest--;
         added++;
      }

      body += "]";
      if(!PostJson(url, body))
      {
         Print("GoldBridgeEA failed sending historical batch.");
         return false;
      }
   }

   Print("GoldBridgeEA historical M1 upload complete. Bars=", copied);
   return true;
}

bool SendLivePacket()
{
   string symbol = BridgeSymbol;
   if(!SymbolSelect(symbol, true)) return false;

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(symbol, PERIOD_M1, 0, 1, rates) != 1) return false;

   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick)) return false;

   double spread = tick.ask - tick.bid;
   string body = PacketJson(symbol, rates[0], tick.bid, tick.ask, spread);
   return PostJson(BridgeUrl, body);
}

int OnInit()
{
   EventSetTimer(1);
   Print("GoldBridgeEA started. Mode=", AccountModeText(), " account=", AccountInfoInteger(ACCOUNT_LOGIN), " symbol=", BridgeSymbol);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   if(!history_sent)
   {
      history_sent = SendHistory();
      if(!history_sent) return;
   }

   long now = TimeLocal();
   if(now - last_send < SendEverySeconds) return;
   last_send = now;
   SendLivePacket();
}

void OnTick() {}
