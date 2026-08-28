#property strict
#property description "GOLD M1 data bridge + human-triggered manual order executor for Gold Scalping AI Lab"

#include <Trade/Trade.mqh>

input string BridgeUrl = "http://127.0.0.1:8765/ingest";
input string BridgeSymbol = "GOLD";
input int SendEverySeconds = 2;
input int HistoricalBars = 500;
input int BatchSize = 100;
input bool EnableManualOrders = true;
input ulong ManualOrderMagic = 26082801;

long last_send = 0;
bool history_sent = false;
CTrade trade;

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

string SafeQuery(string s)
{
   StringReplace(s, " ", "_");
   StringReplace(s, "&", "_");
   StringReplace(s, "?", "_");
   StringReplace(s, "#", "_");
   StringReplace(s, "/", "_");
   StringReplace(s, "|", "_");
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

string BaseUrl()
{
   string u = BridgeUrl;
   int p = StringFind(u, "/ingest");
   if(p >= 0) return StringSubstr(u, 0, p);
   return u;
}

string BatchUrl()
{
   return BaseUrl() + "/ingest_batch";
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
      Print("GoldBridgeEA WebRequest error: ", GetLastError());
      return false;
   }
   if(code < 200 || code >= 300)
   {
      Print("GoldBridgeEA HTTP error: ", code);
      return false;
   }
   return true;
}

bool GetText(string url, string &text)
{
   char data[];
   char result[];
   string response_headers;
   string headers = "Accept: text/plain\r\n";
   ResetLastError();
   int code = WebRequest("GET", url, headers, 3000, data, result, response_headers);
   if(code == -1 || code < 200 || code >= 300) return false;
   text = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
   return true;
}

void PositionSnapshot(string symbol, string &ticket, string &side, double &volume, double &price, double &sl, double &tp)
{
   ticket = "";
   side = "NONE";
   volume = 0.0;
   price = 0.0;
   sl = 0.0;
   tp = 0.0;
   if(!PositionSelect(symbol)) return;
   ticket = IntegerToString((long)PositionGetInteger(POSITION_TICKET));
   long type = PositionGetInteger(POSITION_TYPE);
   side = type == POSITION_TYPE_BUY ? "BUY" : "SELL";
   volume = PositionGetDouble(POSITION_VOLUME);
   price = PositionGetDouble(POSITION_PRICE_OPEN);
   sl = PositionGetDouble(POSITION_SL);
   tp = PositionGetDouble(POSITION_TP);
}

string PacketJson(string symbol, MqlRates &bar, double bid, double ask, double spread)
{
   string pos_ticket, pos_side;
   double pos_volume, pos_price, pos_sl, pos_tp;
   PositionSnapshot(symbol, pos_ticket, pos_side, pos_volume, pos_price, pos_sl, pos_tp);
   return StringFormat(
      "{\"account_id\":\"%I64d\",\"account_mode\":\"%s\",\"broker\":\"%s\",\"symbol\":\"%s\",\"timestamp\":\"%s\",\"open\":%.8f,\"high\":%.8f,\"low\":%.8f,\"close\":%.8f,\"bid\":%.8f,\"ask\":%.8f,\"spread\":%.8f,\"balance\":%.2f,\"equity\":%.2f,\"current_position_ticket\":\"%s\",\"current_position_side\":\"%s\",\"current_position_volume\":%.4f,\"current_position_price\":%.8f,\"current_position_sl\":%.8f,\"current_position_tp\":%.8f}",
      AccountInfoInteger(ACCOUNT_LOGIN), AccountModeText(), JsonEscape(AccountInfoString(ACCOUNT_COMPANY)),
      JsonEscape(symbol), IsoTime(bar.time), bar.open, bar.high, bar.low, bar.close, bid, ask, spread,
      AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY),
      JsonEscape(pos_ticket), JsonEscape(pos_side), pos_volume, pos_price, pos_sl, pos_tp
   );
}

bool SendHistory()
{
   string symbol = BridgeSymbol;
   if(!SymbolSelect(symbol, true)) return false;
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
         if(spread <= 0.0) spread = (double)SymbolInfoInteger(symbol, SYMBOL_SPREAD) * point;
         double bid = bar.close - spread / 2.0;
         double ask = bar.close + spread / 2.0;
         if(added > 0) body += ",";
         body += PacketJson(symbol, bar, bid, ask, spread);
         oldest--;
         added++;
      }
      body += "]";
      if(!PostJson(url, body)) return false;
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
   return PostJson(BridgeUrl, PacketJson(symbol, rates[0], tick.bid, tick.ask, tick.ask - tick.bid));
}

double NormalizeVolume(string symbol, double requested)
{
   double minv = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxv = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0.0) step = minv;
   double v = MathMax(minv, MathMin(maxv, requested));
   v = MathFloor(v / step + 0.5) * step;
   return NormalizeDouble(v, 2);
}

void ReportCommand(string command_id, bool ok, string detail)
{
   string account = StringFormat("%I64d", AccountInfoInteger(ACCOUNT_LOGIN));
   string pos_ticket, pos_side;
   double pos_volume, pos_price, pos_sl, pos_tp;
   PositionSnapshot(BridgeSymbol, pos_ticket, pos_side, pos_volume, pos_price, pos_sl, pos_tp);

   string url = BaseUrl() + "/command_result/" + account + "/" + command_id;
   url += "?ok=" + (ok ? "1" : "0");
   url += "&detail=" + SafeQuery(detail);
   url += "&order_ticket=" + IntegerToString((long)trade.ResultOrder());
   url += "&deal_ticket=" + IntegerToString((long)trade.ResultDeal());
   url += "&executed_price=" + DoubleToString(trade.ResultPrice(), 8);
   url += "&volume=" + DoubleToString(trade.ResultVolume(), 4);
   url += "&retcode=" + IntegerToString((int)trade.ResultRetcode());
   url += "&retcode_description=" + SafeQuery(trade.ResultRetcodeDescription());
   url += "&position_ticket=" + pos_ticket;
   string ignored;
   GetText(url, ignored);
}

void PollManualOrder()
{
   if(!EnableManualOrders) return;
   string account = StringFormat("%I64d", AccountInfoInteger(ACCOUNT_LOGIN));
   string text;
   if(!GetText(BaseUrl() + "/command_text/" + account, text)) return;
   StringTrimLeft(text);
   StringTrimRight(text);
   if(text == "" || text == "NONE") return;

   string parts[];
   int count = StringSplit(text, '|', parts);
   if(count != 6)
   {
      Print("GoldBridgeEA invalid command: ", text);
      return;
   }

   string command_id = parts[0];
   string side = parts[1];
   string symbol = parts[2];
   double volume = NormalizeVolume(symbol, StringToDouble(parts[3]));
   double sl = StringToDouble(parts[4]);
   double tp = StringToDouble(parts[5]);

   if(!SymbolSelect(symbol, true))
   {
      ReportCommand(command_id, false, "symbol_not_available");
      return;
   }

   trade.SetExpertMagicNumber(ManualOrderMagic);
   trade.SetTypeFillingBySymbol(symbol);
   bool ok = false;
   if(side == "BUY") ok = trade.Buy(volume, symbol, 0.0, sl, tp, "AI-Lab manual BUY");
   else if(side == "SELL") ok = trade.Sell(volume, symbol, 0.0, sl, tp, "AI-Lab manual SELL");

   string detail = "retcode_" + IntegerToString((int)trade.ResultRetcode());
   Print("GoldBridgeEA manual order ", side, " id=", command_id, " ok=", ok,
         " retcode=", trade.ResultRetcode(), " ", trade.ResultRetcodeDescription(),
         " order=", trade.ResultOrder(), " deal=", trade.ResultDeal(), " price=", trade.ResultPrice());
   ReportCommand(command_id, ok, detail);
}

int OnInit()
{
   EventSetTimer(1);
   Print("GoldBridgeEA started. Mode=", AccountModeText(), " account=", AccountInfoInteger(ACCOUNT_LOGIN),
         " symbol=", BridgeSymbol, " manualOrders=", EnableManualOrders);
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
   PollManualOrder();
   long now = TimeLocal();
   if(now - last_send < SendEverySeconds) return;
   last_send = now;
   SendLivePacket();
}

void OnTick() {}
