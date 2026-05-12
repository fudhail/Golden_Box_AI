//+------------------------------------------------------------------+
//|                                       GoldenBox_Visualizer.mq5   |
//|                               Copyright 2026, Golden AI Engine   |
//+------------------------------------------------------------------+
#property copyright "Golden AI Engine"
#property link      ""
#property version   "1.00"
#property indicator_chart_window

// We will use 4 buffers:
// 1. Box High Line
// 2. Box Low Line
// 3. Upside Fakeout (Sell Arrow)
// 4. Downside Fakeout (Buy Arrow)

#property indicator_buffers 4
#property indicator_plots   4

//--- plot BoxHigh
#property indicator_label1  "Box High"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//--- plot BoxLow
#property indicator_label2  "Box Low"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrDodgerBlue
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

//--- plot Sell Arrow (Upside Fakeout)
#property indicator_label3  "Upside Fakeout (Sell Scan)"
#property indicator_type3   DRAW_ARROW
#property indicator_color3  clrRed
#property indicator_width3  2

//--- plot Buy Arrow (Downside Fakeout)
#property indicator_label4  "Downside Fakeout (Buy Scan)"
#property indicator_type4   DRAW_ARROW
#property indicator_color4  clrLime
#property indicator_width4  2

//--- input parameters
input int InpBoxWindow = 10;          // Box Rolling Window (Bars)

//--- indicator buffers
double         BufferBoxHigh[];
double         BufferBoxLow[];
double         BufferSellScan[];
double         BufferBuyScan[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0, BufferBoxHigh, INDICATOR_DATA);
   SetIndexBuffer(1, BufferBoxLow, INDICATOR_DATA);
   SetIndexBuffer(2, BufferSellScan, INDICATOR_DATA);
   SetIndexBuffer(3, BufferBuyScan, INDICATOR_DATA);

//--- set arrow codes
   PlotIndexSetInteger(2, PLOT_ARROW, 234); // Down arrow
   PlotIndexSetInteger(3, PLOT_ARROW, 233); // Up arrow

//--- set empty values
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, 0.0);
   PlotIndexSetDouble(3, PLOT_EMPTY_VALUE, 0.0);

   return(INIT_SUCCEEDED);
  }
  
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   // We need at least InpBoxWindow + 1 bars to calculate
   if(rates_total < InpBoxWindow + 1)
      return(0);

   int limit = prev_calculated == 0 ? InpBoxWindow : prev_calculated - 1;

   for(int i = limit; i < rates_total; i++)
     {
      // Calculate Box High / Low from the PREVIOUS 10 candles
      // i is current candle. i-1 to i-10 are previous.
      double box_high = -DBL_MAX;
      double box_low = DBL_MAX;
      
      for(int j = 1; j <= InpBoxWindow; j++)
        {
         if(i - j >= 0)
           {
            if(high[i - j] > box_high) box_high = high[i - j];
            if(low[i - j] < box_low)   box_low = low[i - j];
           }
        }
        
      BufferBoxHigh[i] = box_high;
      BufferBoxLow[i] = box_low;
      
      BufferSellScan[i] = 0.0;
      BufferBuyScan[i] = 0.0;

      // Check for Fakeouts on closed candles (we evaluate the logic at the close of candle)
      // Since live_trader checks the *last closed* candle (index -1 in python, i-1 here if we want closed)
      // But we can just draw it on the candle itself as it forms or closes.
      
      bool upside_fakeout = (high[i] > box_high) && (close[i] < box_high) && (open[i] < box_high);
      bool downside_fakeout = (low[i] < box_low) && (close[i] > box_low) && (open[i] > box_low);
      
      if(upside_fakeout)
        {
         // Draw a down arrow slightly above the high
         BufferSellScan[i] = high[i] + (high[i] - low[i]) * 0.2; 
        }
        
      if(downside_fakeout)
        {
         // Draw an up arrow slightly below the low
         BufferBuyScan[i] = low[i] - (high[i] - low[i]) * 0.2; 
        }
     }
     
   return(rates_total);
  }
//+------------------------------------------------------------------+
