import { defineConfig } from '@playwright/test';
export default defineConfig({testDir:'./tests',workers:1,timeout:30000,use:{baseURL:'http://127.0.0.1:8000',headless:true,launchOptions:{executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'}}});
