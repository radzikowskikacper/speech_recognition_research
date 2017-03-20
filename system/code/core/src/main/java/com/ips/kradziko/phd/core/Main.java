package com.ips.kradziko.phd.core;

import com.ips.kradziko.phd.core.analysis.Alignment;

import java.io.IOException;

/**
 * Created by kradziko on 17/02/15.
 */
public class Main {
    public static void main(String args[]) throws IOException {
        Alignment al = new Alignment();
        al.loadData("/home/kradziko/projects/research/phd/data");
    }
}
