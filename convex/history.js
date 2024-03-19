import { mutation } from "./_generated/server";
import { v } from "convex/values";

export const send = mutation({
    args: { prompt: v.string()  },
    handler: async (ctx, args) => {
      const promptId = await ctx.db.insert("history", { prompt: args.prompt});
      
    },
}); 

