package com.tencent.scrfdncnn;

import android.view.View;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.view.MotionEvent;
import android.view.View;
public class GameView extends View {
    //    声明Paint对象
    private Paint mPaint = null;
    private int StrokeWidth = 5;
    private boolean IsUsed = false;
    private boolean IsClicked = false;
    public Rect rect = new Rect(0,0,0,0);//手动绘制矩形

    public GameView(Context context){
        super(context);
        //构建对象
        mPaint = new Paint();
        mPaint.setColor(Color.RED);
    }
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        //设置无锯齿
        mPaint.setAntiAlias(true);
        //canvas.drawARGB(25, 255, 0, 0);
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeWidth(StrokeWidth);
        //mPaint.setColor(Color.GREEN);
        mPaint.setAlpha(100);
        mPaint.setColor(Color.RED);
        canvas.drawRect(rect,mPaint);
    }
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (IsUsed) return false;
        int x = (int)event.getX();
        int y = (int)event.getY();
        switch (event.getAction()){
            case MotionEvent.ACTION_DOWN:
                rect.right+=StrokeWidth;
                rect.bottom+=StrokeWidth;
                invalidate(rect);
                rect.left = x;
                rect.top = y;
                rect.right =rect.left;
                rect.bottom = rect.top;
                //IsClicked = true;
            case MotionEvent.ACTION_MOVE:
                Rect old = new Rect(rect.left,rect.top,rect.right+StrokeWidth,rect.bottom+StrokeWidth);
                rect.right = x;
                rect.bottom = y;
                old.union(x,y);
                invalidate(old);
                break;

            case MotionEvent.ACTION_UP:
                //if(IsClicked)
                //    IsUsed = true;
                break;
            default:
                break;
        }
        return true;  //未启用时不处理触摸信息
    }
    public void SelectRect() {
        IsUsed = true;
        rect.left = 0;
        rect.top = 0;
        rect.right = 0;
        rect.bottom = 0;
        invalidate(rect);
    }
}
