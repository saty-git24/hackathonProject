import { mutation } from "./_generated/server";
import { v } from "convex/values";

export const send = mutation({
    args: { prompt: v.string(), model_emo: v.string(), user_emo: v.string() },
    handler: async (ctx, args) => {
      const feedbackId = await ctx.db.insert("feedback", 
      { 
        prompt: args.prompt,  
        model_emo: args.model_emo,  
        user_emo: args.user_emo
      });
      console.log(`successfully recorded history: ${feedbackId}`)
    },
}); 
