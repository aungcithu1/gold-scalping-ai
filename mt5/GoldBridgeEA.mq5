#property strict
#property description "Read-only XAUUSD M1 data bridge for Gold Scalping AI Lab"

input string BridgeUrl = "http://127.0.0.1:8765/ingest";
input string BridgeSymbol = "XAUUSD";
input int SendEverySeconds = 2;

long last_send = 0;

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

bool SendPacket()
{
   string symbol = BridgeSymbol;
   if(!SymbolSelect(symbol, true)) return false;

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(symbol, PERIOD_M1, 0, 1, rates) != 1) return false;

   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick)) return false;

   double spread = tick.ask - tick.bid;
   string ts = TimeToString(rates[0].time, TIME_DATE|TIME_MINUTES|TIME_SECONDS);
   StringReplace(ts, ".", "-");
   StringReplace(ts, " ", "T");
   ts += "Z";

   string body = StringFormat(
      "{\"account_id\":\"%I64d\",\"account_mode\":\"%s\",\"broker\":\"%s\",\"symbol\":\"%s\",\"timestamp\":\"%s\",\"open\":%.8f,\"high\":%.8f,\"low\":%.8f,\"close\":%.8f,\"bid\":%.8f,\"ask\":%.8f,\"spread\":%.8f,\"balance\":%.2f,\"equity\":%.2f}",
      AccountInfoInteger(ACCOUNT_LOGIN),
      AccountModeText(),
      JsonEscape(AccountInfoString(ACCOUNT_COMPANY)),
      JsonEscape(symbol),
      ts,
      rates[0].open,
      rates[0].high,
      rates[0].low,
      rates[0].close,
      tick.bid,
      tick.ask,
      spread,
      AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoDouble(ACCOUNT_EQUITY)
   );

   char data[];
   StringToCharArray(body, data, 0, WHOLE_ARRAY, CP_UTF8);
   if(ArraySize(data) > 0) ArrayResize(data, ArraySize(data)-1);
   char result[];
   string response_headers;
   string headers = "Content-Type: application/json\r\n";

   ResetLastError();
   int code = WebRequest("POST", BridgeUrl, headers, 2500, data, result, response_headers);
   if(code == -1)
   {
      Print("GoldBridgeEA WebRequest error: ", GetLastError(), ". Add bridge URL to MT5 allowed WebRequest URLs.");
      return false;
   }
   return code >= 200 && code < 300;
}

int OnInit()
{
   EventSetTimer(1);
   Print("GoldBridgeEA started. Mode=", AccountModeText(), " account=", AccountInfoInteger(ACCOUNT_LOGIN));
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   long now = TimeLocal();
   if(now - last_send < SendEverySeconds) return;
   last_send = now;
   SendPacket();
}

void OnTick() {}
