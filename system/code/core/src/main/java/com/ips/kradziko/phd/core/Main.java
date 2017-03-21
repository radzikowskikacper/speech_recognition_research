package com.ips.kradziko.phd.core;

import com.ips.kradziko.phd.core.data_handler.DataHandler;

import java.io.IOException;

/**
 * Created by kradziko on 17/02/15.
 */
public class Main {
    public static void main(String args[]) throws IOException {
        DataHandler dh = new DataHandler();
        dh.loadData("/home/kradziko/projects/research/phd/data");
        if(true) return;
    }
}
